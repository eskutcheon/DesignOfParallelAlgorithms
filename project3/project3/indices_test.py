import numpy as np

def get_RL_encoding(arr):
    # array of indices in arr where multiple accesses occur during the loop
    accesses_idx = np.where(arr > 1)
    num_accesses = accesses_idx[0].size
    acc_encoded = {}
    acc_set = set()
    i = 0
    while i < num_accesses-1:
        if accesses_idx[0][i] not in acc_set:
            acc_set.add(accesses_idx[0][i])
            run = 1
            while (i+run < num_accesses) and (accesses_idx[0][i+run] == (accesses_idx[0][i] + run)):
                run += 1
            # second element is run length
            acc_encoded[accesses_idx[0][i]] = run-1
            i += run
        else:
            print("might have messed up run length encoding")
    return acc_encoded

def get_indices_list(ni, nj, nk, iskip, jskip, kstart, skip_len, allocsize):
    idx_arr = np.zeros(allocsize, dtype=int)
    for i in range(ni):
        for j in range(nj):
            offset = int(kstart + i*iskip + j*jskip)
            for k in range(nk):
                idx = int(k+offset)
                idx_arr[idx] += 1
                idx_arr[idx-skip_len] += 1
    return idx_arr


def indices_test(idx_list, ni, nj, nk, iskip, jskip, kstart, skip, allocsize):
    count = 0
    num_keys = len(idx_list)
    for i in range(ni):
        for j in range(nj):
            offset = int(kstart + i*iskip + j*jskip)
            for k in range(nk):
                idx = int(k+offset)
                if (count < num_keys) and (idx == idx_list[count]):
                    count += 1
                    print(f'i,j,k = {i,j,k}')



if __name__ == "__main__":
    ni = nj = nk = 32
    allocsize = (ni + 4)*(nj + 4)*(nk + 4)
    iskip = (nk + 4)*(nj + 4)
    jskip = (nk + 4)
    kskip = 1
    kstart = 2*iskip + 2*jskip + 2

    i_faces = get_indices_list(ni+1, nj, nk, iskip, jskip, kstart, iskip, allocsize)
    j_faces = get_indices_list(ni, nj+1, nk, iskip, jskip, kstart, jskip, allocsize)
    k_faces = get_indices_list(ni, nj, nk+1, iskip, jskip, kstart, kskip, allocsize)
    #print(np.sum(i_faces > 1))
    #print(np.sum(j_faces > 1))
    #print(np.sum(k_faces > 1))
    i_RL_dict = get_RL_encoding(i_faces)
    j_RL_dict = get_RL_encoding(j_faces)
    k_RL_dict = get_RL_encoding(k_faces)
    print(len(i_RL_dict.keys()))
    #print(i_RL_dict.keys(), '\n')
    #print(j_RL_dict.keys(), '\n')
    #print(k_RL_dict.keys(), '\n')
    # print(i_RL_dict == j_RL_dict) # true
    indices_test(list(i_RL_dict.keys()), ni+1, nj, nk, iskip, jskip, kstart, iskip, allocsize)
    indices_test(list(j_RL_dict.keys()), ni, nj+1, nk, iskip, jskip, kstart, jskip, allocsize)
    indices_test(list(k_RL_dict.keys()), ni, nj, nk+1, iskip, jskip, kstart, kskip, allocsize)