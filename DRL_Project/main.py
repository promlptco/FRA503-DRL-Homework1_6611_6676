"""
Main entry point for DRL_Project_extra (enhanced, GPU-ready).

Usage:
  python main.py train --experiment C3 --episodes 300000 --num-envs 4
  python main.py train --experiment all --parallel --num-envs 4
  python main.py eval  --log-dir logs
  python main.py play  --mode ai_vs_human --agent checkpoints/C3_best.pt
  python main.py play  --mode ai_vs_human --agent checkpoints/C0_best.pt   # cross-device OK

Cross-device play:
  Copy the checkpoints/ folder from any device (DRL_Project or DRL_Project_extra)
  and pass the .pt file with --agent. The loader auto-detects architecture.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Splendor DRL_extra -- Dueling DDQN + PER + Self-play (GPU-ready)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # Train
    tp = subparsers.add_parser("train")
    tp.add_argument("--experiment",    type=str,   default="C3",
                    choices=["C0", "C1", "C2", "C3", "C4", "C5", "C6", "all"])
    tp.add_argument("--episodes",      type=int,   default=300_000)
    tp.add_argument("--eval-interval", type=int,   default=1_000)
    tp.add_argument("--eval-games",    type=int,   default=100)
    tp.add_argument("--tau",           type=float, default=0.005)
    tp.add_argument("--lr",            type=float, default=1e-3)
    tp.add_argument("--gamma",         type=float, default=0.99)
    tp.add_argument("--batch-size",    type=int,   default=64)
    tp.add_argument("--buffer-size",   type=int,   default=100_000)
    tp.add_argument("--save-dir",      type=str,   default="checkpoints")
    tp.add_argument("--log-dir",       type=str,   default="logs")
    tp.add_argument("--parallel",      action="store_true")
    tp.add_argument("--num-envs",      type=int,   default=1,
                    help="Parallel environments per experiment (4-8 recommended on GPU)")

    # Eval
    ep = subparsers.add_parser("eval")
    ep.add_argument("--log-dir",        type=str, default="logs")
    ep.add_argument("--plot-dir",       type=str, default="plots")
    ep.add_argument("--checkpoint-dir", type=str, default="checkpoints")

    # Play
    pp = subparsers.add_parser("play")
    pp.add_argument("--mode",  type=str,   default="ai_vs_ai",
                    choices=["ai_vs_ai", "ai_vs_human"])
    pp.add_argument("--agent", type=str,   default=None,
                    help="Path to checkpoint (.pt) from any device or project version")
    pp.add_argument("--speed", type=float, default=1.0)

    args = parser.parse_args()

    if args.command == "train":
        from train import train, train_parallel
        experiments = (["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
                       if args.experiment == "all" else [args.experiment])
        shared = dict(
            num_episodes=args.episodes,
            eval_interval=args.eval_interval,
            eval_games=args.eval_games,
            lr=args.lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            tau=args.tau,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            num_envs=args.num_envs,
        )
        if args.parallel and len(experiments) > 1:
            train_parallel(experiments=experiments, **shared)
        else:
            for exp in experiments:
                train(experiment=exp, **shared)

    elif args.command == "eval":
        from evaluate import generate_all_plots
        generate_all_plots(args.log_dir, args.plot_dir,
                           experiments=["C0", "C1", "C2", "C3", "C4", "C5", "C6"],
                           checkpoint_dir=args.checkpoint_dir)

    elif args.command == "play":
        from visualize import run_visualization
        run_visualization(mode=args.mode, agent_path=args.agent, ai_speed=args.speed)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
