import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from collections import deque
import torch

class PerformanceLogger:
    """Logs and visualizes training metrics"""
    
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.rally_records = []
        self.loss_history = []
        self.epsilon_history = []
        
        # Rolling windows
        self.reward_window = deque(maxlen=100)
        self.loss_window = deque(maxlen=100)
        
        # Style setup
        sns.set(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        self.colors = sns.color_palette("husl", 4)
        
        self.start_time = time.time()
    
    def log_episode(self, reward, length, rally, loss, epsilon):
        """Record episode statistics"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.rally_records.append(rally)
        self.loss_history.append(loss)
        self.epsilon_history.append(epsilon)
        
        self.reward_window.append(reward)
        self.loss_window.append(loss)
    
    def get_running_stats(self):
        """Return current rolling averages"""
        return {
            'reward': np.mean(self.reward_window) if self.reward_window else 0,
            'loss': np.mean(self.loss_window) if self.loss_window else 0,
            'time_elapsed': time.time() - self.start_time
        }
    
    def visualize_progress(self, save=True):
        """Generate training progress visualization"""
        plt.clf()
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Reward plot
        ax1.plot(self.episode_rewards, color=self.colors[0], alpha=0.3)
        ax1.plot(self._smooth(self.episode_rewards, 10), color=self.colors[0], linewidth=2)
        ax1.set_title("Training Rewards")
        
        # Loss plot
        ax2.plot(self.loss_history, color=self.colors[1], alpha=0.3)
        ax2.plot(self._smooth(self.loss_history, 10), color=self.colors[1], linewidth=2)
        ax2.set_title("Training Loss")
        
        # Rally plot
        ax3.plot(self.rally_records, color=self.colors[2], alpha=0.3)
        ax3.plot(self._smooth(self.rally_records, 10), color=self.colors[2], linewidth=2)
        ax3.set_title("Rally Length")
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.log_dir / "training_progress.png")
            plt.close()
    
    def save_training_report(self):
        """Save text summary of training"""
        report = [
            f"Training Report - {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Episodes: {len(self.episode_rewards)}",
            f"Best Rally: {max(self.rally_records) if self.rally_records else 0}",
            f"Final Epsilon: {self.epsilon_history[-1]:.4f}" if self.epsilon_history else ""
        ]
        
        with open(self.log_dir / "training_report.txt", "w") as f:
            f.write("\n".join(report))
    
    def _smooth(self, data, window_size):
        """Simple moving average smoothing"""
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    def _format_time(self, seconds):
        return time.strftime("%H:%M:%S", time.gmtime(seconds))