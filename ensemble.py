"""
Run evaluation with saved models.
"""
import random
import argparse
from tqdm import tqdm
import torch

from dataloader.loader import DataLoader
from model.trainer import GCNTrainer
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str,default='saved_models/TAC_02_67.10_baseline_2l_1k',  help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)


model_file_list = ['saved_models/01',
                   'saved_models/02',
                   'saved_models/03',
                   'saved_models/04',
                   'saved_models/00']


prob_list = []
for j in range(len(model_file_list)):
    model_file = model_file_list[j] + '/' + args.model
    print("Loading model from {}".format(model_file))
    opt = torch_utils.load_config(model_file)  # 加载超参数
    trainer = GCNTrainer(opt)  # 定义模型
    trainer.load(model_file)  # 加载最好模型
    # load vocab
    vocab_file = model_file_list[j] + '/vocab.pkl'
    vocab = Vocab(vocab_file, load=True)
    assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

    # load data
    data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
    print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
    batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True)
    helper.print_config(opt)
    label2id = constant.LABEL_TO_ID
    id2label = dict([(v, k) for k, v in label2id.items()])
    predictions = []
    all_probs = []
    losses = 0
    batch_iter = tqdm(batch)
    for i, b in enumerate(batch_iter):
        preds, probs, loss = trainer.predict(b)
        predictions += preds
        all_probs += probs
        losses += loss
    prob_a = np.array(all_probs)
    prob_list.append(prob_a)

#prob_all = (prob_list[0] + prob_list[1] + prob_list[2] + prob_list[3] + prob_list[4])/5
prob_all = (prob_list[0] + prob_list[1] + prob_list[2] + prob_list[3])/4

# pa_tree_file = 'saved_models/other_method/pa_tree_pro.npy'
# pa_tree_pro = np.load(pa_tree_file)
#
# prob_all = 0.25 * pa_tree_pro + 0.75 * prob_all

prob_all = torch.from_numpy(prob_all)

pre_out = np.argmax(prob_all.data.cpu().numpy(), axis=1).tolist()

label_out = [id2label[p] for p in pre_out]

p, r, f1 = scorer.score(batch.gold(), label_out, verbose=True)
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset, p, r, f1))
# losses = losses / len(batch_iter)
# print(losses)

print("Evaluation ended.")
print('a')

