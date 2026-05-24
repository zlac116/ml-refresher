"""Equity research desk assistant package.

Loading this package automatically reads `.env` so any downstream LangChain
import resolves API keys correctly. Do not import third-party packages above
the `load_dotenv()` call.
"""

from dotenv import load_dotenv

load_dotenv()
