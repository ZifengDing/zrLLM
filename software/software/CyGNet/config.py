import  argparse

args = argparse.ArgumentParser()
args.add_argument('--dataset', type=str, default='ACLED')
args.add_argument('--time-stamp', type=int, default=1)
args.add_argument('--lr', type=float, default=0.001)
args.add_argument('--n-epochs', type=int, default=30)
args.add_argument('--hidden_dim', type=int, default=200)
args.add_argument("--gpu", type=int, default=0,
                  help="gpu")
args.add_argument("--regularization", type=float, default=0.01,
                    help="regularization weight")
args.add_argument('--valid-epoch', type=int, default=5)
args.add_argument('--alpha', type=float, default=0)
args.add_argument('--batch-size', type=int, default=512)
args.add_argument('--embedding_dim', type=int, default=200)
args.add_argument('--raw', action='store_true', default=False)
args.add_argument('--time_aware_filter', action='store_true', default=True)
args.add_argument('--LLM_init', action='store_true', default=False)
args.add_argument('--path', action='store_true', default=False)
args.add_argument('--LLM_path', action='store_true', default=False)
args.add_argument('--pure_LLM', action='store_true', default=False)
args.add_argument('--counts', type=int, default=4)
args.add_argument('--entity', type=str, default='subject')
args.add_argument("--path_method", type=str, default='tucker',
                    help="which method used in relation path")
args.add_argument('--gamma_init', type=float, default=0.01, help='initialiation of gamma value')
args.add_argument('--gamma_fix', type=int, default=1, help='whether to fix gamma, specify 0 if unfix')
args.add_argument('--withRegul', action='store_true', default=False)



args = args.parse_args()
print(args)