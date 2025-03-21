# Wandb
import argparse
import wandb
from wandb_sweep import sweep

#  Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# Python imports
import tqdm
from tqdm import tqdm
import os
from os.path import join as ospj
import csv

#Local imports
from data import dataset as dset
from models.common import Evaluator
from utils.utils import save_args, load_args
from utils.config_model import configure_model
from flags import parser, DATA_FOLDER

best_auc = 0
best_hm = 0
compose_switch = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):
    global best_auc, best_hm
    best_auc = 0
    best_hm = 0
    # Get arguments and start logging
    if args == None:
        args = parser.parse_args()
        load_args(args.config, args)

    assert not (args.model=='symnet' and args.fast_eval), "fast_eval is currently not available for SymNet"

    logpath = os.path.join(args.cv_dir, args.name)
    os.makedirs(logpath, exist_ok=True)
    save_args(args, logpath, args.config)
    writer = SummaryWriter(log_dir = logpath, flush_secs = 30)

    # Get dataset
    trainset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase='train',
        split=args.splitname,
        model =args.image_extractor,
        num_negs=args.num_negs,
        pair_dropout=args.pair_dropout,
        update_features = args.update_features,
        train_only= args.train_only,
        open_world=args.open_world
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)

    valset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase='val',
        split=args.splitname,
        model=args.image_extractor,
        subset=args.subset,
        update_features=args.update_features,
        open_world=args.open_world
    )
    valoader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers)
    
    testset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase='test',
        split=args.splitname,
        model =args.image_extractor,
        subset=args.subset,
        update_features = args.update_features,
        open_world=args.open_world
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers)
    # Get model and optimizer
    image_extractor, model, optimizer = configure_model(args, trainset)
    args.extractor = image_extractor

    train = train_normal

    evaluator_val =  Evaluator(valset, model)
    evaluator_test =  Evaluator(testset, model)

    print(model)

    start_epoch = 0
    # Load checkpoint
    if args.load is not None:
        checkpoint = torch.load(args.load)
        if image_extractor:
            try:
                image_extractor.load_state_dict(checkpoint['image_extractor'])
                if args.freeze_features:
                    print('Freezing image extractor')
                    image_extractor.eval()
                    for param in image_extractor.parameters():
                        param.requires_grad = False
            except:
                print('No Image extractor in checkpoint')
        model.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        print('Loaded model from ', args.load)

    if args.is_wandb:
        wandb.watch(model)

    for epoch in tqdm(range(start_epoch, args.max_epochs + 1), desc = 'Current epoch'):
        train(epoch, image_extractor, model, trainloader, optimizer, writer)
        if model.is_open and ((epoch+1)%args.update_feasibility_every)==0:
            if args.model=='compcos':
                print('Updating feasibility scores')
                model.update_feasibility(epoch+1.)
            if args.feasibility_adjacency:
                print('Updating adjacency matrix')
                model.update_adj(epoch + 1.)


        if epoch % args.eval_val_every == 0:
            with torch.no_grad():
                if args.fast_eval:
                    do_test = test_fast(epoch, image_extractor, model, valoader, evaluator_val, writer, args, logpath, 'dev')
                    if do_test:
                        test_fast(epoch, image_extractor, model, testloader, evaluator_test, writer, args, logpath, 'test')
                else:
                    test(epoch, image_extractor, model, testloader, evaluator_val, writer, args, logpath)
    print('Best AUC achieved is ', best_auc)
    print('Best HM achieved is ', best_hm)


def train_normal(epoch, image_extractor, model, trainloader, optimizer, writer):
    '''
    Runs training for an epoch
    '''

    if image_extractor:
        image_extractor.train()
    model.train() # Let's switch to training

    train_loss = 0.0 
    for idx, data in tqdm(enumerate(trainloader), total=len(trainloader), desc = 'Training'):
        data  = [d.to(device) for d in data]

        if image_extractor:
            data[0] = image_extractor(data[0])
        
        loss, _ = model(data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        train_loss += loss.item()

    train_loss = train_loss/len(trainloader)
    writer.add_scalar('Loss/train_total', train_loss, epoch)
    print('Epoch: {}| Loss: {}'.format(epoch, round(train_loss, 2)))


def test(epoch, image_extractor, model, testloader, evaluator, writer, args, logpath):
    '''
    Runs testing for an epoch
    '''
    global best_auc, best_hm

    def save_checkpoint(filename):
        state = {
            'net': model.state_dict(),
            'epoch': epoch,
            'AUC': stats['AUC']
        }
        if image_extractor:
            state['image_extractor'] = image_extractor.state_dict()
        torch.save(state, os.path.join(logpath, 'ckpt_{}.t7'.format(filename)))

    if image_extractor:
        image_extractor.eval()

    model.eval()

    accuracies, all_sub_gt, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], [], [], []

    for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing'):
        data = [d.to(device) for d in data]

        if image_extractor:
            data[0] = image_extractor(data[0])

        _, predictions, _ = model(data)

        attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]

        all_pred.append(predictions)
        all_attr_gt.append(attr_truth)
        all_obj_gt.append(obj_truth)
        all_pair_gt.append(pair_truth)

    if args.cpu_eval:
        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt), torch.cat(all_obj_gt), torch.cat(all_pair_gt)
    else:
        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt).to('cpu'), torch.cat(all_obj_gt).to(
            'cpu'), torch.cat(all_pair_gt).to('cpu')

    all_pred_dict = {}
    # Gather values as dict of (attr, obj) as key and list of predictions as values
    if args.cpu_eval:
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.cat(
                [all_pred[i][k].to('cpu') for i in range(len(all_pred))])
    else:
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.cat(
                [all_pred[i][k] for i in range(len(all_pred))])

    # Calculate best unseen accuracy
    results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=args.bias, topk=args.topk)
    stats = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict, topk=args.topk)

    stats['a_epoch'] = epoch

    result = ''
    # write to Tensorboard
    for key in stats:
        writer.add_scalar(key, stats[key], epoch)
        result = result + key + '  ' + str(round(stats[key], 4)) + '| '

    result = result + args.name
    print(f'Test Epoch: {epoch}')
    print(result)
    # Todo if wandb : skip
    if epoch > 0 and epoch % args.save_every == 0:
        save_checkpoint(epoch)
    if stats['AUC'] > best_auc:
        best_auc = stats['AUC']
        print('New best AUC ', best_auc)
        save_checkpoint('best_auc')

    if stats['best_hm'] > best_hm:
        best_hm = stats['best_hm']
        print('New best HM ', best_hm)
        save_checkpoint('best_hm')

    # Logs
    with open(ospj(logpath, 'logs.csv'), 'a') as f:
        w = csv.DictWriter(f, stats.keys())
        if epoch == 0:
            w.writeheader()
        w.writerow(stats)


def test_fast(epoch, image_extractor, model, testloader, evaluator, writer, args, logpath, split='dev'):
    '''
    Runs testing for an epoch
    '''
    global best_auc, best_hm, best_unseen

    def save_checkpoint(filename, key='AUC'):
        state = {
            'net': model.state_dict(),
            'epoch': epoch,
            key: stats[key]
        }
        if image_extractor:
            state['image_extractor'] = image_extractor.state_dict()
        torch.save(state, os.path.join(logpath, 'ckpt_{}.t7'.format(filename)))

    if image_extractor:
        image_extractor.eval()

    model.eval()


    bias = args.bias
    biaslist = None
    for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Computing bias'):
        data = [d.to(device) for d in data]

        if image_extractor and args.model!='MAT':
            data[0] = image_extractor(data[0])

        scores, _, _ = model(data)
        scores = scores.to('cpu')

        attr_truth, obj_truth, pair_truth = data[1].to('cpu'), data[2].to('cpu'), data[3].to('cpu')

        biaslist = evaluator.compute_biases(scores.to('cpu'), attr_truth, obj_truth, pair_truth, previous_list = biaslist)

    biaslist = list(evaluator.get_biases(biaslist).numpy())
    biaslist.append(bias)



    results = {b: {'unseen':0.,'seen':0.,'total_unseen':0.,'total_seen':0., 'attr_match':0.,'obj_match':0.} for b in biaslist}

    for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing'):
        data = [d.to(device) for d in data]

        if image_extractor:
            data[0] = image_extractor(data[0])

        scores, _, _ = model(data)
        scores = scores.to('cpu')

        attr_truth, obj_truth, pair_truth = data[1].to('cpu'), data[2].to('cpu'), data[3].to('cpu')

        seen_mask = None

        for b in biaslist:
            attr_match, obj_match, seen_match, unseen_match, seen_mask, _ = \
                evaluator.get_accuracies_fast(scores, attr_truth, obj_truth, pair_truth, bias=b, seen_mask=seen_mask)

            results[b]['unseen'] += unseen_match.item()
            results[b]['seen'] += seen_match.item()
            results[b]['total_unseen'] += scores.shape[0]-seen_mask.sum().item()
            results[b]['total_seen'] += seen_mask.sum().item()
            results[b]['attr_match'] += attr_match.item()
            results[b]['obj_match'] += obj_match.item()

    for b in biaslist:
        results[b]['unseen']/= results[b]['total_unseen']
        results[b]['seen']/= results[b]['total_seen']
        results[b]['attr_match']/= (results[b]['total_seen']+results[b]['total_unseen'])
        results[b]['obj_match']/= (results[b]['total_seen']+results[b]['total_unseen'])


    results['a_epoch'] = epoch

    stats = evaluator.collect_results(biaslist,results)

    result = ''
    # write to Tensorboard
    for key in stats:
        writer.add_scalar(key, stats[key], epoch)
        result = result + key + '  ' + str(round(stats[key], 4)) + '| '

    result = result + args.name
    print(f'{split} Epoch: {epoch}')
    print(result)

    if model.args.is_wandb:
        wandb.log({
            split+"_bestAUC": stats['AUC'],
            split+"_bestHM": stats['best_hm']
        })
        if stats['AUC'] > best_auc and split=='dev':
            print("new best model on dev! testing.....")
            best_auc = stats['AUC']
            return True
        else:
            return False

    else:
        if stats['best_hm'] > best_hm:
            best_hm = stats['best_hm']
            print('New best HM ', best_hm)
            save_checkpoint('best_hm')

        
        if stats['AUC'] > best_auc and split=='dev':
            best_auc = stats['AUC']
            print('New best AUC ', best_auc)
            save_checkpoint('best_auc')

        # save_checkpoint(str(epoch)+'_epoch_model')
        
        # Logs
        with open(ospj(logpath, 'logs.csv'), 'a') as f:
            w = csv.DictWriter(f, stats.keys())
            if epoch == 0:
                w.writeheader()
            w.writerow(stats)


def sub():
    with wandb.init(config=None):
        args = wandb.config
        main(args)


def sweep_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', dest='model_name', type=str)
    sweep_args = parser.parse_args()
    
    sweep_config = sweep(sweep_args.model_name)
    sweep_id = wandb.sweep(sweep_config, project=sweep_args.model_name) # 'tobesuperior/cgqa_unmatch/f65cjv4t' # wandb.sweep(sweep_config, project=sweep_args.model_name)
    wandb.agent(sweep_id, sub, count=100)


if __name__ == '__main__':
    try:
        main(None)
        # sweep_config()
    except KeyboardInterrupt:
        print('Best AUC achieved is ', best_auc)
        print('Best HM achieved is ', best_hm)