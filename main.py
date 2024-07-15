import re
import sys
from streamlit.web import cli as stcli

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "bot.py"]
    sys.exit(stcli.main())
