from math import sqrt
import numpy as np
from scipy.optimize import curve_fit

def euclidian_distance(x1, y1, x2, y2):

    euc_dis = sqrt(((x1-x2)**2) + ((y1-y2)**2))

    return euc_dis


def y_from_x (x_list,init_speed, theta,gravity):
    return x_list*(np.tan(theta)-0.5*gravity*x_list/(init_speed*np.cos(theta))**2) 

def projectile_fill(queue):
    x_list=[]
    y_list=[]
    for i in range(len(queue)-10,len(queue)):
        x_list.append(queue[i][0])
        y_list.append(queue[i][1])

    intial_speed=0.05
    gravity=9.81/(25*25)
    theta=0.5235  

    average_speed=(x_list[-1]-x_list[-3])/2

    print(average_speed)

    popt_x,pcov_x= curve_fit(y_from_x,x_list,y_list,p0=[intial_speed,theta,gravity],maxfev=5000000)


    intial_speed_x_popt,theta_x_popt,gravity_popt=popt_x

    prev_point=x_list[-1]

    x_filled= prev_point+average_speed
    y_filled= y_from_x(x_filled,intial_speed_x_popt,theta_x_popt,gravity_popt)

    return x_filled, y_filled
    


def falsepositive_removal_with_filling(que,boxes, no_miss_detect):
    
    queue=que.copy()
    if len(queue)<10:
        for box in boxes:
            if len(box) !=0:
                centre_x = box[0] + box[2]//2
                centre_y = box[1] + box[3]//2
                queue.append([centre_x,centre_y])
        
        return queue,no_miss_detect,2

    if len(boxes)==0 :
        if no_miss_detect>5:
            print('case 1, Same returned')
            # no_miss_detect=no_miss_detect+1
            return queue, no_miss_detect,0

        elif no_miss_detect<=5:
            print('case 2, filled')
            print(len(boxes))
            x_filled,y_filled= projectile_fill(queue)
            print(x_filled, y_filled)
            queue.append([int(x_filled),int(y_filled)])
            return queue, no_miss_detect,1

    elif len(boxes)>0:
        if no_miss_detect>5:
            print('case 3')
            dist_compent=5-no_miss_detect
        elif no_miss_detect<=5:
            print('case 4')
            dist_compent=1

        correct_detect=[]
        for box in boxes:
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            # print('This is box: ', box)

            # print("the cords are these :",x0,y0,x1,y1)

            x_test=(x0+x1)//2
            y_test=(y0+y1)//2

            print('These are x and y test',x_test,y_test)

            x_filled,y_filled= projectile_fill(queue)

            print('These are the filled points :', x_filled, y_filled)

            distance=euclidian_distance(x_test,y_test,x_filled,y_filled)

            dist_thres=50 

            if distance<= dist_thres*dist_compent:
                correct_detect.append([x_test,y_test])
            else:
                print(distance, dist_thres*dist_compent)
                print("False detect")

        
        correct_detect=np.array(correct_detect)
        if len(correct_detect)>=1:
            x_final=np.sum(correct_detect,axis=0)[0]//len(boxes)
            y_final=np.sum(correct_detect,axis=0)[1]//len(boxes)

            queue.append([x_final,y_final])
            print('Case 3a and 4a mids returned')
            return queue, no_miss_detect,2
        
        elif len(correct_detect)==0:
            if no_miss_detect>5:
                print('case 3 b, same queue returned')
                no_miss_detect=no_miss_detect+1
                return queue, no_miss_detect,0
            elif no_miss_detect<=5:
                print('case 4 b, Filled')
                queue.append([int(x_filled),int(y_filled)])
                return queue,no_miss_detect,1






                

                





    








    


    




