import numpy as np
import matplotlib.pyplot as plt


def smooth_reward(reward, integrator=15) : 
    """ Returns the episode reward suitable for plotting by integrating over 
        the last n values as specified on the plot
        
    """
    y = []
    for i in range(len(reward) - integrator):
        temp = np.sum(reward[i:i+integrator])/integrator
        y.append(temp)
        
    x = np.arange(len(y))
    return np.array([x,y])

color_scheme = {'logs_noisy_vector.csv':'blue',
                'logs_noisy_vec_img.csv':'green',
                'logs_noisy_image.csv' : 'tab:blue',
                'logs_noisy_img_vec.csv' : 'tab:orange',
                'logs_vector.csv':'pink',
                'logs_noisy_vec_noisy_img.csv':'black',
                'logs_vector_visual.csv':'red',
                'logs_image.csv':'brown'}

folder = 'data/aiming_vector/'

file_1 = 'logs_noisy_vector.csv'
file_2 = 'logs_noisy_image.csv'
file_3 = 'logs_noisy_vec_noisy_img.csv'
file_4 = 'logs_vector_visual.csv'
file_5 = 'logs_image.csv'
file_6 =  'logs_vector.csv'
file_7 =  'logs_noisy_vec_img.csv'
file_8 =  'logs_noisy_img_vec.csv'

curve_1 = np.genfromtxt(folder+file_1)[1:]
reward_1 = smooth_reward(curve_1[:,3])
plt.plot(reward_1[0],reward_1[1],label='Noisy vector only',c=color_scheme[file_1])

curve_2 = np.genfromtxt(folder+file_2)[1:]
reward_2 = smooth_reward(curve_2[:,3])
plt.plot(reward_2[0],reward_2[1], label='Noisy image only',c=color_scheme[file_2])

curve_3 = np.genfromtxt(folder+file_3)[1:]
reward_3 = smooth_reward(curve_3[:,3])
plt.plot(reward_3[0],reward_3[1], label='Noisy vector and image',c=color_scheme[file_3])

curve_4 = np.genfromtxt(folder+file_4)[1:]
reward_4 = smooth_reward(curve_4[:,3])
plt.plot(reward_4[0],reward_4[1], label='Vector and image',c=color_scheme[file_4])

curve_5 = np.genfromtxt(folder+file_5)[1:]
reward_5 = smooth_reward(curve_5[:,3])
plt.plot(reward_5[0],reward_5[1], label='Image only',c=color_scheme[file_5])

curve_6 = np.genfromtxt(folder+file_6)[1:]
reward_6 = smooth_reward(curve_6[:,3])
plt.plot(reward_6[0],reward_6[1], label='Vector only',c=color_scheme[file_6])

curve_7 = np.genfromtxt(folder+file_7)[1:]
reward_7 = smooth_reward(curve_7[:,3])
plt.plot(reward_7[0],reward_7[1], label='Noisy vector+visual',c=color_scheme[file_7])

curve_8 = np.genfromtxt(folder+file_8)[1:]
reward_8 = smooth_reward(curve_8[:,3])
plt.plot(reward_8[0],reward_8[1], label='Noisy visual + vector',c=color_scheme[file_8])



plt.xlabel('trials')
plt.ylabel('reward')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),ncol=2)
plt.tight_layout()