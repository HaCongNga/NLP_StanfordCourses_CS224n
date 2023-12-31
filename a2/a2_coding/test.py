import numpy as np
a = np.array([1,2,3])
a = a.T
print(a)
print(a.shape)
print(np.sum(a))
b = np.array([1, 2, 3]).T
print(np.dot(a,b))
#a = [[1,2], [2,3]]

print(np.dot(3, a))



u_o = outsideVectors[outsideWordIdx]
u_k = outsideVectors[negSampleWordIndices]
v_c = centerWordVec

loss = -np.log(sigmoid(np.dot(u_o.T, v_c))) - np.sum(np.log(sigmoid(-np.dot(u_k, v_c))))

gradCenterVec = -(1-sigmoid(np.dot(u_o.T, v_c))) * u_o - np.dot(-(1-sigmoid(-np.dot(u_k, v_c))).T, u_k)

gradOutsideVecs = np.zeros(outsideVectors.shape)
gradOutsideVecs[outsideWordIdx, :] = (sigmoid(np.dot(u_o.T, v_c)) - 1) * v_c
unique, indices, counts = np.unique(negSampleWordIndices, return_index=True, return_counts=True)
sigmoidUniqueSamplesCenter = sigmoid(-np.dot(u_k, v_c))[indices]
gradOutsideVecs[unique, :] = - ((sigmoidUniqueSamplesCenter - 1) * counts)[:, None] * v_c[None, :]

gradOutsideVecs new: [[ 0.72782715 -0.65644417 -0.54611399]
 [-0.17872052  0.16119217  0.13410021]
 [ 0.49813754 -0.44928179 -0.3737699 ]
 [ 1.12120276 -1.01123875 -0.84127737]
 [ 0.57369411 -0.51742802 -0.43046262]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]

gradOutsideVecs old: [[ 1.68180414 -1.51685813 -1.26191606]
 [-0.17872052  0.16119217  0.13410021]
 [ 0.4852181  -0.43762945 -0.36407599]
 [ 0.38246274 -0.34495201 -0.28697508]
 [ 0.72782715 -0.65644417 -0.54611399]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]

 other_prob_dis: [-0.25954923 -0.25954923  1.12920814  1.43339694  1.12920814  1.09428521
  1.12920814  1.09428521  1.43339694  1.43339694]
other_prob_dis[unique] : [0 2 3 4], [-0.25954923  1.12920814  1.43339694  1.12920814]
other_prob_dis[unique]_prob : [0 2 3 4], [0.56452548 0.24430727 0.19256995 0.24430727]
other_prob_dis[unique]*counts : [3 2 2 3], [0 2 3 4], [1.69357644 0.48861453 0.38513991 0.7329218 ]
centerWordVec: [ 0.99304885 -0.89565377 -0.74511904]
gradOutsideVecs: [[ 1.68180414 -1.51685813 -1.26191606]
 [-0.17872052  0.16119217  0.13410021]
 [ 0.4852181  -0.43762945 -0.36407599]
 [ 0.38246274 -0.34495201 -0.28697508]
 [ 0.72782715 -0.65644417 -0.54611399]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
gradOutsideVecs[unique] : [[ 1.68180414 -1.51685813 -1.26191606]
 [ 0.4852181  -0.43762945 -0.36407599]
 [ 0.38246274 -0.34495201 -0.28697508]
 [ 0.72782715 -0.65644417 -0.54611399]]
gradOutsideVecs right: [[ 0.72782715 -0.65644417 -0.54611399]
 [-0.17872052  0.16119217  0.13410021]
 [ 0.49813754 -0.44928179 -0.3737699 ]
 [ 1.12120276 -1.01123875 -0.84127737]
 [ 0.57369411 -0.51742802 -0.43046262]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]

iter 0 , neg_index 3, prob 0.5645254797597057, centerWord: [ 0.99304885 -0.89565377 -0.74511904], grad: [ 0.56060138 -0.50561938 -0.42063869]
iter 1 , neg_index 3, prob 0.5645254797597057, centerWord: [ 0.99304885 -0.89565377 -0.74511904], grad: [ 1.12120276 -1.01123875 -0.84127737]     
iter 2 , neg_index 0, prob 0.24430726551612303, centerWord: [ 0.99304885 -0.89565377 -0.74511904], grad: [ 0.24260905 -0.21881472 -0.182038  ]    
iter 3 , neg_index 4, prob 0.19256995262638932, centerWord: [ 0.99304885 -0.89565377 -0.74511904], grad: [ 0.19123137 -0.17247601 -0.14348754]    
iter 4 , neg_index 0, prob 0.24430726551612303, centerWord: [ 0.99304885 -0.89565377 -0.74511904], grad: [ 0.4852181  -0.43762945 -0.36407599]    
iter 5 , neg_index 2, prob 0.2508122040907179, centerWord: [ 0.99304885 -0.89565377 -0.74511904], grad: [ 0.24906877 -0.2246409  -0.18688495]     
iter 6 , neg_index 0, prob 0.24430726551612303, centerWord: [ 0.99304885 -0.89565377 -0.74511904], grad: [ 0.72782715 -0.65644417 -0.54611399]
iter 7 , neg_index 2, prob 0.2508122040907179, centerWord: [ 0.99304885 -0.89565377 -0.74511904], grad: [ 0.49813754 -0.44928179 -0.3737699 ]     
iter 8 , neg_index 4, prob 0.19256995262638932, centerWord: [ 0.99304885 -0.89565377 -0.74511904], grad: [ 0.38246274 -0.34495201 -0.28697508]    
iter 9 , neg_index 4, prob 0.19256995262638932, centerWord: [ 0.99304885 -0.89565377 -0.74511904], grad: [ 0.57369411 -0.51742802 -0.43046262] 