from math import log
import numpy as np
import matplotlib.pyplot as plt

def get_func_flops(ni, nj, nk, f, func_name):
    print(f"flop count of {func_name}: {f(ni,nj,nk)}")
    return f(ni,nj,nk)

def count_total_flops(n, f_dict):
    sum_1 = 0
    for val in f_dict.values():
        sum_1 += val(n)
    return sum_1

# get dictionary of asymptotic arithmetic intensity values for each function
def get_AI_dict(n, flop_dict, byte_dict):
    AI_dict = {}
    num_keys = len(flop_dict.keys())
    arr_sum = 0
    for key in flop_dict.keys():
        AI_dict[key] = flop_dict[key](n)/byte_dict[key](n)
        print(f"{key}:\n flops={flop_dict[key](n)}\n bytes={byte_dict[key](n)}\n AI={AI_dict[key]}")
        arr_sum += AI_dict[key]
    print(arr_sum/num_keys)
    return AI_dict

if __name__ == "__main__":
    peak_flops = 2496*0.706*2 # 2496 cuda cores * .706 Ghz * 2 Flops/cycle for Gigaflops
    bandwidth = 208 # Memory bandwidth: 208 GB/sec
    # dict of the exact flop count functions in each kernel function
    ker_ex_flops = {"setInitialConditions" : lambda ni, nj, nk: 6*ni*nk + 57*ni*nj*nk,
                    "copyPeriodic" : lambda ni, nj, nk: 189*ni*nk,
                    "zeroResidual" : lambda ni, nj, nk: 5*(ni+2)*(nj+2)*(nk+2),
                    "computeResidual" : lambda ni, nj, nk: 3*(2 + (120)*(ni+1)*nj*nk),
                    "computeStableTimestep" : lambda ni, nj, nk: 31*ni*nj*nk,
                    "minKernel" : lambda ni, nj, nk: int((2e-10 + 2)*ni*nk + log(nk, 2)-1
                                                        + sum([(ni*nj)/(2**m) for m in range(1, int(log(nk, 2)))])),
                    "integrateKineticEnergy" : lambda ni, nj, nk: 4*ni*nk + 13*ni*nj*nk,
                    "sumKernel" : lambda ni, nj, nk: int((2e-10 + 2)*ni*nk + log(nk, 2)-1
                                                        + sum([(ni*nj)/(2**m) for m in range(1, int(log(nk, 2)))])),
                    "weightedSum" : lambda ni, nj, nk: 10*ni*nj*nk}

    ker_asym_flops = {"setInitialConditions" : lambda n: 57*n**3,
                    "copyPeriodic" : lambda n: 189*n**2,
                    "zeroResidual" : lambda n: 5*n**3,
                    "computeResidual" : lambda n: 120*n**3, # for just a single kernel
                    "computeStableTimestep" : lambda n: 31*n**3,
                    "minKernel" : lambda n: (n**2)*log(n, 2),
                    "integrateKineticEnergy" : lambda n: 13*n**3,
                    "sumKernel" : lambda n: (n**2)*log(n, 2),
                    "weightedSum" : lambda n: 10*n**3}

    ker_asym_bytes = {"setInitialConditions" : lambda n: 4*54*n**3,
                    "copyPeriodic" : lambda n: 4*341*n**2,
                    "zeroResidual" : lambda n: 4*38*n**3,
                    "computeResidual" : lambda n: 4*176*n**3, # for just a single kernel
                    "computeStableTimestep" : lambda n: 4*47*n**3,
                    "minKernel" : lambda n: 4*2*(n**2)*log(n, 2),
                    "integrateKineticEnergy" : lambda n: 4*25*n**3,
                    "sumKernel" : lambda n: 4*2*(n**2)*log(n, 2),
                    "weightedSum" : lambda n: 4*20*n**3}

    # flop_counts = [count_total_flops(32*m, ker_asym_flops) for m in [1, 2, 4, 8]]
    AI_list = []
    #print(f"flop count for ni=nj=nk=32*mult: (exact count, asymptotic count)")
    for i in range(4):
        #print(f"flop count for ni=nj=nk={32*(2**i)}: {flop_counts[i]}")
        AI_list.append(get_AI_dict(32*(2**i), ker_asym_flops, ker_asym_bytes))
        #print(AI_list[i])
        print('\n')

    # will be the same for all mesh sizes due to asymptotic term dominance
    intersection = peak_flops/bandwidth
    x_bound = 1.5*intersection # to include more of the roofline intersection
    xx_sloped = np.linspace(0, intersection, 1000)
    bandwidth_bound = 208*xx_sloped
    performance_bound = (peak_flops)*np.ones(1000)
    xx_flat = np.linspace(intersection, x_bound, 1000)
    plt.plot(xx_sloped, bandwidth_bound, 'b', label = 'peak bandwidth performance')
    plt.plot(xx_flat, performance_bound, 'r', label = 'peak arithmetic performance')
    plt.plot(intersection*np.ones(100), np.linspace(0, peak_flops, 100), '--k')
    plt.legend()
    plt.grid()
    plt.xlabel('Arithmetic Intensity (flops/byte)')
    plt.ylabel('Arithmetic Performance (Gigaflops/s)')
    plt.title('Roofline Plot for Parallel Fluid Simulation')
    plt.show()

