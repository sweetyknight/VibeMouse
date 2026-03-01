from __future__ import annotations

from vibemouse.app import VoiceMouseApp
from vibemouse.config import load_config


def main() -> None:
    config = load_config()
    app = VoiceMouseApp(config)
    app.run()


if __name__ == "__main__":
    main()
