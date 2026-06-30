import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm
import json

from .config import ModelConfig
from .data_loader import CryptoDataLoader
from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .vm import StackVM
from .backtest import MemeBacktest
from .ops import OPS_CONFIG
from .vocab import FORMULA_VOCAB

class AlphaEngine:
    def __init__(self, use_lord_regularization=True, lord_decay_rate=1e-3, lord_num_iterations=5):
        """
        Initialize AlphaGPT training engine.
        
        Args:
            use_lord_regularization: Enable Low-Rank Decay (LoRD) regularization
            lord_decay_rate: Strength of LoRD regularization
            lord_num_iterations: Number of Newton-Schulz iterations per step
        """
        self.loader = CryptoDataLoader()
        self.loader.load_data()
        
        self.model = AlphaGPT().to(ModelConfig.DEVICE)
        
        # Standard optimizer
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=ModelConfig.LEARNING_RATE)
        
        # Low-Rank Decay regularizer
        self.use_lord = use_lord_regularization
        if self.use_lord:
            self.lord_opt = NewtonSchulzLowRankDecay(
                self.model.named_parameters(),
                decay_rate=lord_decay_rate,
                num_iterations=lord_num_iterations,
                target_keywords=["q_proj", "k_proj", "attention", "qk_norm"]
            )
            self.rank_monitor = StableRankMonitor(
                self.model,
                target_keywords=["q_proj", "k_proj"]
            )
        else:
            self.lord_opt = None
            self.rank_monitor = None
        
        self.vm = StackVM()
        self.bt = MemeBacktest()
        self.operator_offset = FORMULA_VOCAB.operator_offset
        self.vocab_size = FORMULA_VOCAB.size
        self.arity_tensor = torch.zeros(self.vocab_size, dtype=torch.long, device=ModelConfig.DEVICE)
        for i, cfg in enumerate(OPS_CONFIG):
            self.arity_tensor[self.operator_offset + i] = cfg[2]
        self.max_stack_reduction = max(cfg[2] - 1 for cfg in OPS_CONFIG)
        self.walk_forward_slices = self._build_walk_forward_slices(self.loader.target_ret.shape[1])
        
        self.best_score = -float('inf')
        self.best_formula = None
        self.training_history = {
            'step': [],
            'avg_reward': [],
            'avg_val_score': [],
            'best_score': [],
            'valid_rate': [],
            'actor_loss': [],
            'critic_loss': [],
            'entropy': [],
            'stable_rank': []
        }

    def _build_walk_forward_slices(self, total_steps):
        usable_steps = max(1, total_steps - 2)
        train_end = max(1, int(usable_steps * ModelConfig.WALK_FORWARD_TRAIN_RATIO))
        if train_end >= usable_steps:
            return [(slice(0, usable_steps), slice(0, usable_steps))]

        remaining = usable_steps - train_end
        fold_count = max(1, min(ModelConfig.WALK_FORWARD_FOLDS, remaining))
        val_size = max(1, remaining // fold_count)

        slices = []
        for fold_idx in range(fold_count):
            val_start = train_end + fold_idx * val_size
            if val_start >= usable_steps:
                break
            val_end = usable_steps if fold_idx == fold_count - 1 else min(usable_steps, val_start + val_size)
            if val_end > val_start:
                # Expanding train window followed by the next forward validation window.
                slices.append((slice(0, val_start), slice(val_start, val_end)))

        return slices or [(slice(0, train_end), slice(train_end, usable_steps))]

    def _can_finish_formula(self, stack_depth, steps_remaining):
        reduce_needed = torch.clamp(stack_depth - 1, min=0)
        min_reduce_steps = torch.div(
            reduce_needed + self.max_stack_reduction - 1,
            self.max_stack_reduction,
            rounding_mode="floor",
        )
        return (stack_depth >= 1) & (min_reduce_steps <= steps_remaining)

    def _build_action_mask(self, stack_depth, step):
        mask = torch.full(
            (stack_depth.shape[0], self.vocab_size),
            float("-inf"),
            device=ModelConfig.DEVICE,
        )
        steps_remaining = ModelConfig.MAX_FORMULA_LEN - step - 1

        for token_id in range(self.vocab_size):
            if token_id < self.operator_offset:
                new_depth = stack_depth + 1
                has_args = torch.ones_like(stack_depth, dtype=torch.bool)
            else:
                arity = self.arity_tensor[token_id]
                has_args = stack_depth >= arity
                new_depth = stack_depth - arity + 1

            allowed = has_args & self._can_finish_formula(new_depth, steps_remaining)
            mask[allowed, token_id] = 0.0

        return mask

    def _update_stack_depth(self, stack_depth, action):
        arity = self.arity_tensor[action]
        is_feature = action < self.operator_offset
        delta = torch.where(is_feature, torch.ones_like(stack_depth), 1 - arity)
        return stack_depth + delta

    def _evaluate_walk_forward(self, factors):
        train_scores = []
        val_scores = []
        val_returns = []

        for train_slice, val_slice in self.walk_forward_slices:
            train_score, _ = self.bt.evaluate(
                factors,
                self.loader.raw_data_cache,
                self.loader.target_ret,
                time_slice=train_slice,
            )
            val_score, val_ret = self.bt.evaluate(
                factors,
                self.loader.raw_data_cache,
                self.loader.target_ret,
                time_slice=val_slice,
            )
            train_scores.append(train_score)
            val_scores.append(val_score)
            val_returns.append(val_ret)

        train_score = torch.stack(train_scores).mean()
        val_score = torch.stack(val_scores).mean()
        policy_score = train_score + ModelConfig.VALIDATION_REWARD_WEIGHT * val_score
        avg_val_ret = sum(val_returns) / len(val_returns)
        return policy_score, val_score, avg_val_ret

    def train(self):
        print("🚀 Starting Meme Alpha Mining with LoRD Regularization..." if self.use_lord else "🚀 Starting Meme Alpha Mining...")
        if self.use_lord:
            print(f"   LoRD Regularization enabled")
            print(f"   Target keywords: ['q_proj', 'k_proj', 'attention', 'qk_norm']")
        print(f"   Walk-forward folds: {len(self.walk_forward_slices)}")
        
        pbar = tqdm(range(ModelConfig.TRAIN_STEPS))
        
        for step in pbar:
            bs = ModelConfig.BATCH_SIZE
            inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)
            stack_depth = torch.zeros(bs, dtype=torch.long, device=ModelConfig.DEVICE)
            
            log_probs = []
            values = []
            entropies = []
            tokens_list = []
            
            for gen_step in range(ModelConfig.MAX_FORMULA_LEN):
                logits, value, _ = self.model(inp)
                action_mask = self._build_action_mask(stack_depth, gen_step)
                dist = Categorical(logits=logits + action_mask)
                action = dist.sample()
                
                log_probs.append(dist.log_prob(action))
                values.append(value.squeeze(-1))
                entropies.append(dist.entropy())
                tokens_list.append(action)
                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
                stack_depth = self._update_stack_depth(stack_depth, action)
            
            seqs = torch.stack(tokens_list, dim=1)
            
            rewards = torch.zeros(bs, device=ModelConfig.DEVICE)
            val_scores = torch.full((bs,), -10.0, device=ModelConfig.DEVICE)
            valid_count = 0
            
            for i in range(bs):
                formula = seqs[i].tolist()
                
                res = self.vm.execute(formula, self.loader.feat_tensor)
                
                if res is None:
                    rewards[i] = -5.0
                    continue
                
                if res.std() < 1e-4:
                    rewards[i] = -2.0
                    continue
                
                score, val_score, ret_val = self._evaluate_walk_forward(res)
                rewards[i] = score
                val_scores[i] = val_score
                valid_count += 1
                
                if val_score.item() > self.best_score:
                    self.best_score = val_score.item()
                    self.best_formula = formula
                    tqdm.write(
                        f"[!] New King: WF {score:.2f} | Val {val_score:.2f} | "
                        f"ValRet {ret_val:.2%} | Formula {formula}"
                    )
            
            log_prob_sum = torch.stack(log_probs, dim=1).sum(dim=1)
            value_pred = torch.stack(values, dim=1).mean(dim=1)
            entropy_mean = torch.stack(entropies, dim=1).mean()

            returns = rewards.detach()
            advantage = returns - value_pred.detach()
            adv = (advantage - advantage.mean()) / (advantage.std() + 1e-5)

            actor_loss = -(log_prob_sum * adv).mean()
            critic_loss = F.mse_loss(value_pred, returns)
            entropy_loss = -ModelConfig.ENTROPY_COEF * entropy_mean
            loss = actor_loss + ModelConfig.VALUE_LOSS_COEF * critic_loss + entropy_loss
            
            # Gradient step
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), ModelConfig.GRAD_CLIP_NORM)
            self.opt.step()
            
            # Apply Low-Rank Decay regularization
            if self.use_lord:
                self.lord_opt.step()
            
            # Logging
            avg_reward = rewards.mean().item()
            avg_val_score = val_scores.mean().item()
            valid_rate = valid_count / bs
            postfix_dict = {
                'AvgRew': f"{avg_reward:.3f}",
                'AvgVal': f"{avg_val_score:.3f}",
                'Valid': f"{valid_rate:.1%}",
                'Ent': f"{entropy_mean.item():.2f}",
                'BestVal': f"{self.best_score:.3f}",
            }
            
            if self.use_lord and step % 100 == 0:
                stable_rank = self.rank_monitor.compute()
                postfix_dict['Rank'] = f"{stable_rank:.2f}"
                self.training_history['stable_rank'].append(stable_rank)
            
            self.training_history['step'].append(step)
            self.training_history['avg_reward'].append(avg_reward)
            self.training_history['avg_val_score'].append(avg_val_score)
            self.training_history['best_score'].append(self.best_score)
            self.training_history['valid_rate'].append(valid_rate)
            self.training_history['actor_loss'].append(actor_loss.item())
            self.training_history['critic_loss'].append(critic_loss.item())
            self.training_history['entropy'].append(entropy_mean.item())
            
            pbar.set_postfix(postfix_dict)

        # Save best formula
        strategy = {
            "formula": self.best_formula,
            "vocab_size": FORMULA_VOCAB.size,
            "features": list(FORMULA_VOCAB.feature_names),
            "operators": list(FORMULA_VOCAB.operator_names),
            "best_validation_score": self.best_score,
            "max_formula_len": ModelConfig.MAX_FORMULA_LEN,
        }
        with open("best_meme_strategy.json", "w") as f:
            json.dump(strategy, f, indent=2)
        
        # Save training history
        import json as js
        with open("training_history.json", "w") as f:
            js.dump(self.training_history, f)
        
        print(f"\n✓ Training completed!")
        print(f"  Best score: {self.best_score:.4f}")
        print(f"  Best formula: {self.best_formula}")


if __name__ == "__main__":
    eng = AlphaEngine(use_lord_regularization=True)
    eng.train()
