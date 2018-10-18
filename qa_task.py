import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import pickle
import time
import sys
import os
import beam_search

sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')
from vdnc import VariationalDNC as DNC
from recurrent_controller import StatelessRecurrentController
import data_util
import plot_tool
import random
random.seed(time.time())
TOTAL_ANNEL_EPOCH=100

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def load(path):
    return pickle.load(open(path, 'rb'))



def single_qa_task(args):
    dirname = os.path.dirname(os.path.abspath(__file__)) + '/data/save/'
    print(dirname)
    ckpts_dir = os.path.join(dirname, 'checkpoints_qa_{}_single_in_single_out_persit'.format(args.task))

    llprint("Loading Data ... ")

    llprint("Done!\n")
    str2tok, tok2str, dialogs = pickle.load(open(args.data_dir, 'rb'))

    all_index = list(range(len(dialogs)))
    train_index = all_index[:int(len(dialogs) - args.valid_size*2)]
    valid_index = all_index[int(len(dialogs) - args.valid_size):int(len(dialogs) * 1)]
    test_index = all_index[int(len(dialogs) - args.valid_size*2):int(len(dialogs) -args.valid_size)]

    dialogs_list_train = [dialogs[i] for i in train_index]

    dialogs_list_valid = [dialogs[i] for i in valid_index]

    dialogs_list_test = [dialogs[i] for i in test_index]

    print('num_dialogs {}'.format(len(dialogs)))
    print('num train {}'.format(len(dialogs_list_train)))
    print('num valid {}'.format(len(dialogs_list_valid)))
    print('num test {}'.format(len(dialogs_list_test)))
    print('dim in  {} {}'.format(len(str2tok), len(str2tok)))
    print('dim out {}'.format(len(str2tok)))

    batch_size = args.batch_size
    input_size = len(str2tok)
    output_size = len(str2tok)

    words_count = args.mem_size
    word_size = args.word_size

    learning_rate = args.learning_rate
    momentum = 0.9

    iterations = args.iterations
    start_step = 0


    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = args.cpu_num
    config.inter_op_parallelism_threads = args.cpu_num

    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = args.gpu_ratio
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph, config=config) as session:

            llprint("Building Computational Graph ... ")

            ncomputer = DNC(
                StatelessRecurrentController,
                input_size,
                output_size,
                output_size,
                words_count,
                word_size,
                1,
                batch_size,
                use_mem=args.use_mem,
                dual_emb=False,
                use_emb_encoder=True,
                use_emb_decoder=True,
                decoder_mode=True,
                emb_size=args.emb_dim,
                hidden_controller_dim=args.hidden_dim,
                use_teacher=args.use_teacher,
                attend_dim=args.attend,
                enable_drop_out=args.drop_out_keep>0,
                memory_read_heads_decode=args.num_mog_mode,
                nlayer=args.nlayer,
                name='VDNC',
                gt_type=args.gt_type,
                single_KL=args.single_KL,
                KL_anneal=args.anneal_KL
            )
            optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)
            # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            _, prob, loss, apply_gradients, loss_rec, loss_kl, alpha = \
                ncomputer.build_vloss_function_mask(optimizer, clip_s=10, total_epoch=TOTAL_ANNEL_EPOCH)
            llprint("Done!\n")
            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())
            llprint("Done!\n")

            if args.from_checkpoint is not '':
                if args.from_checkpoint=='default':
                    from_checkpoint = ncomputer.print_config()
                else:
                    from_checkpoint = args.from_checkpoint
                llprint("Restoring Checkpoint %s ... " % from_checkpoint)
                ncomputer.restore(session, ckpts_dir, from_checkpoint)
                llprint("Done!\n")
                mat = None
                if args.use_pretrain_emb=='word2vec':
                    mat = data_util.load_word2vec(emb_dim=args.emb_dim, str2tok_dir=str2tok, init_zero=True)

                elif args.use_pretrain_emb=='glove':
                    mat = data_util.loadGloVe(emb_dim=50, str2tok_dir=str2tok)
            elif args.use_pretrain_emb=='glove':
                mat = data_util.loadGloVe(emb_dim=50, str2tok_dir=str2tok)
                ncomputer.assign_pretrain_emb_encoder(session,
                                                      mat)
                ncomputer.assign_pretrain_emb_decoder(session,
                                                      mat)

            elif args.use_pretrain_emb == 'word2vec':
                mat = data_util.load_word2vec(emb_dim=args.emb_dim, str2tok_dir=str2tok)
                ncomputer.assign_pretrain_emb_encoder(session,
                                                      mat)
                ncomputer.assign_pretrain_emb_decoder(session,
                                                      mat)

            last_100_losses = []
            last_100_losses_rc = []
            last_100_losses_kl = []

            start = 1 if start_step == 0 else start_step + 1
            end = start_step + iterations + 1
            if args.mode == 'test' or args.mode == 'cherry_pick':
                start=0
                end = start
                dialogs_list_valid = dialogs_list_test
            elif args.mode == 'test_file':
                start = 0
                end = start
                dialogs_list_valid = data_util.load_lines_from_file(args.test_file, str2tok)

            start_time_100 = time.time()

            avg_100_time = 0.
            avg_counter = 0
            if args.mode=='train':
                log_dir = './data/summary/log_{}_{}/'.format(args.task, args.use_pretrain_emb)
                if not os.path.isdir(log_dir):
                    os.mkdir(log_dir)
                log_dir = '{}/{}/'.format(log_dir,ncomputer.print_config())
                if not os.path.isdir(log_dir):
                    os.mkdir(log_dir)
                train_writer = tf.summary.FileWriter(log_dir, session.graph)
            min_tloss=0
            alpha_v = 0
            itersave=0
            for i in range(start, end + 1):
                try:
                    llprint("\rIteration %d/%d" % (i, end))
                    input_data, target_output, seq_len, decoder_length, masks,_, _ = \
                        data_util.prepare_sample_batch(dialogs_list_train, input_size, output_size, batch_size)

                    summerize = (i % args.valid_time == 0)
                    if args.mode == 'train':
                        loss_value, loss_vrec, loss_vkl, alpha_v, _ = session.run([
                            loss, loss_rec, loss_kl, alpha,
                            apply_gradients
                        ], feed_dict={
                            ncomputer.input_encoder: input_data,
                            ncomputer.input_decoder: target_output,
                            ncomputer.target_output: target_output,
                            ncomputer.sequence_length: seq_len,
                            ncomputer.decode_length: decoder_length,
                            ncomputer.mask: masks,
                            ncomputer.teacher_force: ncomputer.get_bool_rand_incremental(decoder_length),
                            ncomputer.drop_out_keep: args.drop_out_keep,
                            ncomputer.testing_phase:False,
                            ncomputer.epochs:float(i*args.batch_size*(1.0/args.ratio_start_anneal)//len(dialogs_list_train))
                        })

                        last_100_losses.append(loss_value)
                        last_100_losses_rc.append(loss_vrec)
                        last_100_losses_kl.append(loss_vkl)


                    tloss=10000000
                    tpre=0
                    if summerize:
                        print('start summarize...')
                        llprint("\n\t episode %d -->Avg. Cross-Entropy: %.7f\n" % (i, np.mean(last_100_losses)))
                        print('avg.rec.loss: {}, avg.kl: {}. l.alpha: {}'.format(np.mean(last_100_losses_rc), np.mean(last_100_losses_kl), alpha_v))
                        trscores = []

                        if args.mode=='train':
                            summary = tf.Summary()
                            summary.value.add(tag='batch_train_loss', simple_value=np.mean(last_100_losses))
                            summary.value.add(tag='batch_train_recloss', simple_value=np.mean(last_100_losses_rc))
                            summary.value.add(tag='batch_train_kl', simple_value=np.mean(last_100_losses_kl))
                            for ii in range(5):
                                input_data, target_output, seq_len, decoder_length,masks, brout, brin = \
                                    data_util.prepare_sample_batch(dialogs_list_train, input_size, output_size,
                                                                           batch_size)

                                out, mem_view = session.run([prob, ncomputer.packed_memory_view_decoder],  feed_dict={
                                    ncomputer.input_encoder: input_data,
                                    ncomputer.input_decoder: target_output,
                                    ncomputer.target_output: target_output,
                                    ncomputer.sequence_length: seq_len,
                                    ncomputer.decode_length: decoder_length,
                                    ncomputer.mask: masks,
                                    ncomputer.teacher_force: ncomputer.get_bool_rand_incremental(decoder_length),
                                    ncomputer.drop_out_keep: args.drop_out_keep,
                                    ncomputer.testing_phase:False,
                                    ncomputer.epochs:float(i*args.batch_size*(1.0/args.ratio_start_anneal)//len(dialogs_list_train))
                                                })

                                out = np.reshape(np.asarray(out),[-1, decoder_length, output_size])
                                out = np.argmax(out, axis=-1)
                                bout_list = []
                                for b in range(out.shape[0]):
                                    out_list = []
                                    for io in range(out.shape[1]):
                                        if out[b][io]==2:
                                            break
                                        out_list.append(out[b][io])
                                    bout_list.append(out_list)

                                trscores.append(data_util.bleu_score(np.asarray(brin), np.asarray(brout),
                                                                             np.asarray(bout_list), tok2str))


                            estr = ''
                            dstr = ''
                            for t in brin[0]:
                                estr+=tok2str[t]+' '
                            for tt in range(len(bout_list[0])):
                                print(mem_view['dist1s'][0][tt])
                                print(mem_view['dist2s'][0][tt])
                                print(mem_view['mixturews'][0][tt])
                                print(mem_view['last_reads'][0][tt])
                                print('---')
                                dstr+=tok2str[bout_list[0][tt]]+' '
                                print('{}-->{}'.format(estr,dstr))
                                # plot_tool.plot_mgauss(mem_view['dist2s'][0][tt],
                                #                       mem_view['mixturews'][0][tt],
                                #                       mem_view['dist1s'][0][tt])
                                plot_tool.plot_tsne(mem_view['dist2s'][0][tt],
                                                      mem_view['mixturews'][0][tt],
                                                      mem_view['dist1s'][0][tt])

                            print('+++')

                            print('done quick test train...')

                        tescores = []
                        tescores4 = []
                        distinct2 = []
                        bows = []
                        losses = []
                        losses2=[]
                        losses3 = []
                        all_out=[]
                        all_label=[]
                        all_res_in = []
                        all_res_out = []
                        all_res_pred = []
                        all_res_score = []
                        ntb = len(dialogs_list_valid) // batch_size + 1
                        for ii in range(ntb):
                            # llprint("\r{}/{}".format(ii, ntb))
                            if ii * batch_size == len(dialogs_list_valid):
                                break
                            bs = [ii * batch_size, min((ii + 1) * batch_size, len(dialogs_list_valid))]
                            rs = bs[1] - bs[0]
                            if bs[1] >= len(dialogs_list_valid):
                                bs = [len(dialogs_list_valid) - batch_size, len(dialogs_list_valid)]

                            input_data, target_output, seq_len, decoder_length, masks, rout_list, rin_list = \
                                data_util.prepare_sample_batch(dialogs_list_valid, input_size, output_size, bs)
                            out, loss_v, lost_v_rec, loss_v_kl, mem_view = session.run([prob, loss, loss_rec, loss_kl, ncomputer.packed_memory_view_decoder],
                                                                             feed_dict={ncomputer.input_encoder: input_data,
                                                                               ncomputer.input_decoder: target_output,
                                                                               ncomputer.target_output: target_output,
                                                                               ncomputer.sequence_length: seq_len,
                                                                               ncomputer.decode_length: decoder_length,
                                                                               ncomputer.mask: masks,
                                                                               ncomputer.teacher_force: ncomputer.get_bool_rand_incremental(decoder_length, prob_true_max=0),
                                                                               ncomputer.drop_out_keep: 1,
                                                                               ncomputer.testing_phase: True,
                                                                               ncomputer.epochs:1.0*TOTAL_ANNEL_EPOCH
                                                                               })

                            # print(np.max(mem_view['zs'][0], axis=1))
                            # print(np.min(mem_view['zs'][0], axis=1))
                            # print(np.mean(mem_view['zs'][0], axis=1))

                            # print('---')
                            losses.append(lost_v_rec)
                            losses2.append(loss_v_kl)
                            losses3.append(loss_v)

                            pout = np.reshape(np.asarray(out), [-1, decoder_length, output_size])
                            out = np.argmax(pout, axis=-1)
                            bout_list = []

                            for b in range(rs):
                                if args.beam_size == 0:
                                    out_list = []
                                    for io in range(out.shape[1]):
                                        if out[b][io]==2:
                                            break
                                        out_list.append(out[b][io])

                                else:
                                    out_list = beam_search.leap_beam_search(pout[b],
                                                                            beam_size=args.beam_size,
                                                                            is_set=True, is_fix_length=False, stop_char=2)
                                bout_list.append(out_list)
                            tescores.append(
                                data_util.bleu_score(np.asarray(rin_list)[:rs], np.asarray(rout_list)[:rs],
                                                             np.asarray(bout_list)[:rs], tok2str))
                            if args.mode == 'test':
                                tescores4.append(
                                    data_util.bleu_score4(np.asarray(rin_list)[:rs], np.asarray(rout_list)[:rs],
                                                                  np.asarray(bout_list)[:rs], tok2str, print_prob=0.8))
                                bows.append(data_util.bow_score(np.asarray(rin_list)[:rs],
                                                                                  np.asarray(rout_list)[:rs],
                                                                                  np.asarray(bout_list)[:rs], tok2str, mat))
                            elif args.mode=='cherry_pick':
                                res_in, res_out, res_pred, res_score =data_util.cherry_pick(np.asarray(rin_list)[:rs],
                                                                                  np.asarray(rout_list)[:rs],
                                                                                  np.asarray(bout_list)[:rs], tok2str, 5, mat)
                                all_res_in.extend(res_in)
                                all_res_out.extend(res_out)
                                all_res_pred.extend(res_pred)
                                all_res_score.extend(res_score)

                            all_out+=bout_list[:rs]
                            all_label+=rout_list[:rs]

                        if args.mode == 'test_file':
                            print('some predic')
                            print(len(all_out))
                            print(len(all_label))
                            for tt, tv in enumerate(all_out):
                                # print('{} vs {}'.format(dialogs_list_valid[tt][0], all_out[tt]))
                                str1=''
                                for c in dialogs_list_valid[tt][0]:
                                    str1+=tok2str[c]+' '
                                str2 = ''
                                for c in all_out[tt]:
                                    str2 += tok2str[c] + ' '
                                print('{} --> {}'.format(str1,str2))
                                print('---')
                        elif args.mode == 'cherry_pick':
                            print('=======================')
                            alls = np.asarray(all_res_score)
                            # print(alls)
                            mind = alls.argsort()[::-1][:50]
                            for indd in mind:
                                print('{} --> {} vs {} with score {}'.format(all_res_in[indd], all_res_out[indd],
                                                                             all_res_pred[indd],
                                                                             all_res_score[indd]))

                        tloss=np.mean(losses)
                        tloss2 = np.mean(losses2)
                        tpre=np.mean(tescores)
                        print('tr score {} vs te store {}'.format(np.mean(trscores),tpre))
                        print('kl train {} vs kl test {}'.format(np.mean(last_100_losses_kl),tloss2))
                        if args.mode=='test':
                            tescores4=np.asarray(tescores4)
                            te4 = np.mean(tescores4,axis=0)
                            print('4 bleu')
                            print(te4)
                            print(np.mean(bows))
                        print('test loss {}'.format(tloss))
                        if args.mode=='train':
                            summary.value.add(tag='train_acc', simple_value=np.mean(trscores))
                            summary.value.add(tag='test_acc', simple_value=np.mean(tescores))
                            summary.value.add(tag='test_recloss', simple_value=tloss)
                            summary.value.add(tag='test_kl', simple_value=tloss2)
                            summary.value.add(tag='test_loss', simple_value=np.mean(losses3))
                            train_writer.add_summary(summary, i)
                            train_writer.flush()

                        end_time_100 = time.time()
                        elapsed_time = (end_time_100 - start_time_100) / 60
                        avg_counter += 1
                        avg_100_time += (1. / avg_counter) * (elapsed_time - avg_100_time)
                        estimated_time = (avg_100_time * ((end - i) / 100.)) / 60.

                        print ("\tAvg. 100 iterations time: %.2f minutes" % (avg_100_time))
                        print ("\tApprox. time to completion: %.2f hours" % (estimated_time))

                        start_time_100 = time.time()
                        last_100_losses = []



                        if i>args.min_iter_save and args.mode=='train' and tpre>min_tloss:
                            min_tloss=tpre
                            itersave = i
                            llprint("\nSaving Checkpoint ... "),
                            ncomputer.save(session, ckpts_dir, ncomputer.print_config())
                            llprint("Done!\n")
                        elif i>args.min_iter_save:
                            print('not save as cur loss {} < best {} at step {}'.format(tpre,min_tloss, itersave))
                        else:
                            print('not save as cur iter {} < min save inter {}'.format(i, args.min_iter_save))

                except KeyboardInterrupt:
                    sys.exit(0)



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="train")
    parser.add_argument('--use_mem', default=True, type=str2bool)
    parser.add_argument('--use_teacher', default=False, type=str2bool)
    parser.add_argument('--task', default="cornell")
    parser.add_argument('--data_dir', default="./data/cornell20_20000_10/trim_20qa_single.pkl")
    parser.add_argument('--from_checkpoint', default="")
    parser.add_argument('--hidden_dim', default=768, type=int)
    parser.add_argument('--emb_dim', default=96, type=int)
    parser.add_argument('--attend', default=0, type=int)
    parser.add_argument('--mem_size', default=16, type=int)
    parser.add_argument('--word_size', default=64, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_mog_mode', default=4, type=int)
    parser.add_argument('--beam_size', default=0, type=int)
    parser.add_argument('--nlayer', default=3, type=int)
    parser.add_argument('--drop_out_keep', default=-1, type=float)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--iterations', default=1000000, type=int)
    parser.add_argument('--valid_time', default=100, type=int)
    parser.add_argument('--gpu_ratio', default=0.4, type=float)
    parser.add_argument('--cpu_num', default=10, type=int)
    parser.add_argument('--min_iter_save', default=2000, type=int)
    parser.add_argument('--gpu_device', default="1,2,3", type=str)
    parser.add_argument('--use_pretrain_emb', default="word2vec", type=str)
    parser.add_argument('--gt_type', default="rnn", type=str)
    parser.add_argument('--single_KL', default=False, type=str2bool)
    parser.add_argument('--anneal_KL', default=True, type=str2bool)
    parser.add_argument('--ratio_start_anneal', default=1.0, type=float)
    parser.add_argument('--valid_size', default=10000, type=int)
    parser.add_argument('--test_file', default="./data/test_file.txt", type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    #
    args.mode='train'
    # args.from_checkpoint = 'default'
    # args.use_mem=False
    # args.single_KL=True
    args.num_mog_mode=3
    args.mem_size=15
    # args.batch_size=16
    args.use_pretrain_emb=""
    args.valid_time=1
    args.batch_size=1
    args.valid_size=1
    # args.beam_size = 3
    # args.attend=64
    # args.task = 'cornell20_x2'

    print(args)
    if args.sampled_loss_dim > 0:
        SAMPLED_SOFTMAX = 1


    single_qa_task(args)
