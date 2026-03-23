from __future__ import annotations

import argparse
import traceback

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy2_hfm_live_bot import HfmFixedUsdLiveBot, LiveBotSettings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Strategy 2 HFM live bot (fixed USD per active asset only)."
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to the .env file to load before reading environment variables.",
    )
    args = parser.parse_args()

    settings = LiveBotSettings.from_env(args.env_file)
    bot = HfmFixedUsdLiveBot(settings)

    try:
        bot.run()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        return 130
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
