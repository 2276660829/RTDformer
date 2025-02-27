# from data_provider.data_loader import StockDataset,StockDataset_pred

from data_provider.data_loader_CSI300 import StockDataset
from torch.utils.data import DataLoader

data_dict = {
    'StockDataset' :StockDataset,
}


def data_provider(args, flag,print_debug):
    Data = data_dict[args.data]
    
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1
        freq = args.freq
        num_workers = 0  # 在测试时设置为0
        Data = StockDataset
        print_debug=True
        
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq #"d"
        Data = StockDataset_pred_long
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  #32
        freq = args.freq    #d
        num_workers=args.num_workers

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,  #S
        target=args.target,  #target='Close'
        timeenc=timeenc,    #1
        freq=freq  ,     #d
        num_stock=args.num_stock
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=1,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last)
    return data_set, data_loader
