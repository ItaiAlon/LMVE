import pandas as pd
import argparse

def main(db_name, total, q=0.9):
    assert total > 0

    data = []
    for i in range(total):
        file_name = f'./results/sub/{db_name}/{i}.q.{q}.pd.pkl'
        data.append(pd.read_pickle(file_name))
    df = pd.concat(data, ignore_index=True)
    pd.to_pickle(df, f'./results/{db_name}.q.{q}.pd.pkl')
    print(f'{db_name}.q.{q}.pd.pkl created')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', type=str, help="", required=True)
    parser.add_argument('-total', type=int, help="", required=True)
    parser.add_argument('-q', type=float, default=0.9, required=False)
    args = parser.parse_args()
    main(db_name=args.file, total=args.total, q=args.q)
    print(f'merged {args.file} ({args.total})')
