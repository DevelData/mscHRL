import os
import argparse
import torch as T


def transfer_hidden_weights(source_dir, 
                            target_dir, 
                            save_dir, 
                            params=['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']):
    target_dir = target_dir
    target_nets = os.listdir(target_dir)
    target_nets = [net for net in target_nets if net.endswith(".pth")]
    params = params

    source_dir = source_dir
    source_nets = os.listdir(source_dir)
    source_nets = [net for net in source_nets if net.endswith(".pth")]
    
    device = "cuda:0" if T.cuda.is_available() else "cpu"

    for net in source_nets:
        print(net)
        transfer_net = T.load(target_dir + net)
        source_net = T.load(source_dir + net)

        for param in params:
            print(transfer_net[param].shape, param)
            print("-------------------")
            if (len(transfer_net[param].shape) > 1) and (transfer_net[param].shape[1] > source_net[param].shape[1]):
                transfer_net[param] = T.cat([source_net[param].clone().to(device), T.rand(transfer_net[param].shape[0], transfer_net[param].shape[1] - source_net[param].shape[1]).to(device)], axis=1)
                
            elif (len(transfer_net[param].shape) > 1) and (transfer_net[param].shape[1] < source_net[param].shape[1]):
                transfer_net[param] = source_net[param][:, :transfer_net[param].shape[1]].clone()            
            
            else:
                transfer_net[param] = source_net[param].clone()

        T.save(transfer_net, save_dir + net)
    
    return


parser = argparse.ArgumentParser(description="Transfer weights of networks from source to target directory")

parser.add_argument("-s", "--source_dir", type=str, metavar="", required=True, help="Directory of the networks whose parameters will be transferred.")
parser.add_argument("-t", "--target_dir", type=str, metavar="", required=True, help="Directory of the networks to which the parameters will be transferred.")
parser.add_argument("-S", "--save_dir", type=str, metavar="", required=True, help="Directory in which to save the output.")
args = parser.parse_args()


if __name__ == "__main__":
    transfer_hidden_weights(source_dir=args.source_dir, target_dir=args.target_dir, save_dir=args.save_dir)
    print("Transfer done successfully!")