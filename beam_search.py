import numpy as np


def leap_beam_search_top(out, beam_size=3):
    nstep=out.shape[0]
    nout=out.shape[1]
    pind = np.argsort(out, axis=-1)
    top_labels=[]
    top_probs=[]
    for i in range(nstep):
        c=1
        while c<=beam_size:
            label = pind[i][-c]
            if label not in top_labels:
                top_labels.append(label)
                top_probs.append(out[i][label])
            c+=1

    unorder_predict = [x for _, x in sorted(zip(top_probs, top_labels))]
    return unorder_predict

def leap_beam_search(out, beam_size=3, is_set=False,
                     is_fix_length=False, stop_char=0):
    nstep=out.shape[0]
    nout=out.shape[1]

    C=[]
    smallest_prob_in_c=0
    B={0:[[0,1.0]]}
    pind = np.argsort(out, axis=-1)
    t=0
    while len(B[t])>0 and t<nstep:
        npaths=[]

        for path_prob in B[t]:
            path = path_prob[:-1]
            prob = path_prob[-1]
            c = 1
            while c<=nout:
                label = pind[t][-c]
                if not is_set or label not in path:
                    best_label=pind[t][-c]
                    new=path+[best_label]
                    nprop=out[t][best_label]*prob
                    npaths.append(new+[nprop])
                c+=1
                if len(npaths)==beam_size:
                    break

        sort_npaths=sorted(npaths, key = lambda x: float(x[-1]))
        B[t + 1]=[]
        for k in range(beam_size):
            B[t+1].append(sort_npaths[-k-1])

        for p in B[t+1]:
            if p[-2]==stop_char or (is_fix_length and t+1==nstep):
                C.append(p)
                smallest_prob_in_c = min(smallest_prob_in_c,p[-1])
        new_B=[]
        for p in B[t+1]:
            if p[-1]>smallest_prob_in_c:
                new_B.append(p)
        # print(new_B)
        B[t+1]=new_B
        t=t+1

    if not C:
        return []

    sort_C = sorted(C, key=lambda x: float(x[-1]))
    return sort_C[-1][1:-1]

if __name__ == '__main__':
    a=[[0.1,0.2,0.3,0.4],
       [0.1,0.5,0.2,0.2],
       [0.1,0.8,0.0,0.1]]

    print(leap_beam_search(np.asarray(a), is_set=True, is_fix_length=True))

