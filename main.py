from game.core import PongGame
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Rainbow Pong Training')
    parser.add_argument('--headless', action='store_true', help='Run without visualization')
    parser.add_argument('--episodes', type=int, default=0, help='Max number of episodes to train')
    parser.add_argument('--load', type=str, default='', help='Path to load model weights')
    parser.add_argument('--save', type=str, default='', help='Path to save model weights')
    return parser.parse_args()

def main():
    args = parse_args()
    game = PongGame(headless=args.headless)
    
    if args.load:
        game.agent.model.load_state_dict(torch.load(args.load))
        print(f"Loaded model weights from {args.load}")
    
    try:
        game.run(max_episodes=args.episodes)
    except KeyboardInterrupt:
        print("\nTraining interrupted - saving final results...")
    finally:
        if args.save:
            torch.save(game.agent.model.state_dict(), args.save)
            print(f"Saved model weights to {args.save}")
        print("Training completed!")

if __name__ == "__main__":
    main()