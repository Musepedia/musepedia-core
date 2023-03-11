import argparse

from src.common.log.BaseLogging import init_logger
from src.rpc.server.Server import serve

parser = argparse.ArgumentParser()
parser.add_argument('--qa', action='store_true')
parser.add_argument('--no-qa', dest='qa', action='store_false')
parser.add_argument('--gpt', action='store_true')
parser.add_argument('--no-gpt', dest='gpt', action='store_false')

args = parser.parse_args()

init_logger()
serve(load_qa=args.qa, load_gpt=args.gpt)
