import litserve as ls
from app.api import RAGAPI


def main():

    server = ls.LitServer(
        RAGAPI(),
        spec=ls.OpenAISpec(),
        stream=True,
        timeout=180,
    )

    server.run(port=8001)


if __name__ == "__main__":
    main()