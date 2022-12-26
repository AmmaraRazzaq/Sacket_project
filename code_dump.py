                                # debug_img = np.full((h, w, 3), (255,255,255), dtype='uint8')
                                # pts = np.array(goal_points)
                                # pts = pts.reshape((-1, 1, 2))
                                # cv2.polylines(debug_img, [pts], isClosed=True, color=(0,0,0), thickness=4)
                                # cv2.line(debug_img, (line_pts[0], line_pts[1]), (line_pts[2], line_pts[3]), (255,0,0), 2)
                                # cv2.circle(debug_img, (centre_x, centre_y), radius=2, color=(0,0,0), thickness=2)
                                # cv2.putText(debug_img, txt, (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
                                # debug_img = cv2.resize(debug_img,(810,540),interpolation = cv2.INTER_LINEAR)
                                # cv2.imshow('debug_img', debug_img)
                                # cv2.waitKey(1)

                                    # the ball has gone on the other side of the line
                                    # check the intersection of the ball traj with the last line and if the intersection lies inside the goal post or not
                                    # trajectory line is being considered at an interval of 5 frames
                                    # while doing calculation at each frame, we might be seeing the traj points on the same side of the line.
                                    # so take the last known point before the ball crossed the line 


                                    # # last trajectory point before the ball crossed the line
                                    # cv2.circle(debug_img, (traj_pt[0], traj_pt[1]), radius=2, color=(0,0,0), thickness=8)
                                    # # goal post
                                    # pts = np.array(goal_points)
                                    # pts = pts.reshape((-1, 1, 2))
                                    # cv2.polylines(debug_img, [pts], isClosed=True, color=(0,0,0), thickness=4)
                                    # # last line
                                    # cv2.line(debug_img, (line_pts[0], line_pts[1]), (line_pts[2], line_pts[3]), (255,0,0), 2)
                                    # # traj line
                                    # cv2.line(debug_img, (traj_points[-1][0], traj_points[-1][1]), (traj_pt[0], traj_pt[1]), (0,255,0), 2)
                                    # # intersection point
                                    # cv2.circle(debug_img, (int(intersect_x), int(intersect_y)), radius=2, color=(0,0,255), thickness=4)
                                    # debug_img = cv2.resize(debug_img,(810,540),interpolation = cv2.INTER_LINEAR)
                                    # cv2.imshow('debug_img', debug_img)
                                    # cv2.waitKey(0)

                                    # debugging, draw goal post and ball centre point
                                    # debug_img = np.full((h, w, 3), (255,255,255), dtype='uint8')
                                    # pts = np.array(goal_points)
                                    # pts = pts.reshape((-1, 1, 2))
                                    # cv2.polylines(debug_img, [pts], isClosed=True, color=(0,0,0), thickness=4)
                                    # cv2.line(debug_img, (line_pts[0], line_pts[1]), (line_pts[2], line_pts[3]), (255,0,0), 2)
                                    # cv2.circle(debug_img, (centre_x, centre_y), radius=2, color=(0,0,0), thickness=2)
                                    # cv2.putText(debug_img, txt, (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
                                    # debug_img = cv2.resize(debug_img,(810,540),interpolation = cv2.INTER_LINEAR)
                                    # cv2.imshow('debug_img', debug_img)
                                    # cv2.waitKey(1)
                                    
                                    
                                    # if is_in_goalpost:
                                    #     text = "it is a Goal"
                                    
                                    # if is_in_goalpost or cross_location in ["right", "left", "inside"]:
                                    #     cv2.putText(im0, text, (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
                                    #     cv2.putText(template, text, (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)


##################################################### GMS ###############################################################################################

def gms(line_pts, goal_points,):                        
    print("len(line_pts): ", len(line_pts))
    print("len(goal_points: ", len(goal_points))

    # if the line is visible
    if len(line_pts) == 4:
        line_flag = ball_vs_line(line_pts[:2], line_pts[2:], centre_pt)
        line_flags.append(line_flag)
        print("line_flag: ", line_flag)

        # debug_img = np.full((h, w, 3), (255,255,255), dtype='uint8')
        
        # if line_flag:
        #     traj_pt = pt
        
        # calculate intersection first time the ball crosses the line
        if not line_flag and len(goal_points)==4 and not intersection:
            intersection=True
            intersect_x, intersect_y = line_intersection(traj_points[-1], traj_points[-5], goal_points[0], goal_points[3])
            cross_location = traj_line_intersection((intersect_x,intersect_y), goal_points)                                   
        
        # the ball has crossed the line, keep checking the situation of the ball as the ball location is getting updated in each frame
        if intersection:
            if cross_location == 'right' or cross_location == 'left':
                print("cross location ", cross_location)
                # keep checking if the ball stays outside the line
                line_flag = ball_vs_line(line_pts[:2], line_pts[2:], centre_pt)
                if not line_flag:
                    decision = cross_location
                else:
                    decision = "ball came back"
                
            elif cross_location == 'inside':
                print("cross_location inside")
                # keep checking if the ball stays in the goal post
                is_in_goalpost = check(goal_points[0][0], goal_points[0][1], goal_points[1][0], goal_points[1][1], goal_points[2][0], goal_points[2][1], goal_points[3][0], goal_points[3][1], centre_x, centre_y)
                if is_in_goalpost:
                    decision = "It is a Goal"
                else:
                    decision = "no intersection with goalpost"
            else:
                print("unknown cross location")
                decision = "unknown"
            
            cross_locations.append(decision)
        
        else:
            # the ball has not crossed the line yet 
            # if the goal post is visible, well the line is visible, so that means the goal post is visible
            # check if the trajectory of the ball is towards the goal post, calculate the trajectory line intersection with the last line
            if len(traj_points) >= 5:
                intersect_x, intersect_y = line_intersection(traj_points[-1], traj_points[-5], goal_points[0], goal_points[3])
                cross_location = traj_line_intersection((intersect_x,intersect_y), goal_points) 
                if cross_location == "inside":
                    # the ball is probably moving towards the goal post
                    # check the distance of the latest traj point from the intersection point 
                    distance = euclidean_distance(traj_points[-1][0], traj_points[-1][1], intersect_x, intersect_y)
                    # check if this distance is increasing or decreasing
                    # if distance is decreasing 
                    if distance < prev_distance:
                        prev_distance = distance
                        decreasing_distance_counter += 1 
                    elif distance > prev_distance:
                        prev_distance = distance
                        decreasing_distance_counter = 0 # reset the counter

                    if decreasing_distance_counter >= 25:
                        # if the ball is moving towards the goal post for continuous 25 frames
                        # start looking for deflection in the ball, check when the ball starts moving away
                        if distance > prev_distance:
                            # the ball is now moving away from the goal
                            prev_distance = distance
                            increasing_distance_counter += 1
                        else:
                            prev_distance = distance
                            increasing_distance_counter = 0 # reset the counter
                    
                    if increasing_distance_counter >= 10:
                        # it is most probably a shot on target
                        decision = "Shot on Target"
                        cross_locations.append(decision)

            # if trajectory of the ball was detected for atleast two frames, check the distance of the latest point from the goal post, this distance should decrease for consecutive 25 frames

    # once the line stops being visible, and the goal post stops being visible, but the ball had already crossed the line and some decisions were made, make the final decision
    elif len(line_pts)!=4 and intersection and len(cross_locations)!=0 and len(goal_points)==0:
        if not decision_made:
            final_decision = cross_locations[-1]

            if final_decision in ["right", "left"]:
                final_decision = "It is a Shot Off Target"
            elif final_decision == "no intersection with goalpost":
                final_decision = "It is Shot on target"
            
            cv2.putText(template, final_decision, (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(im0, final_decision, (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4, cv2.LINE_AA)
            template = cv2.resize(template,(810,540),interpolation = cv2.INTER_LINEAR)
            im0 = cv2.resize(im0,(810,540),interpolation = cv2.INTER_LINEAR)
            output_canvas = np.concatenate((template, im0), axis=1)
            cv2.imshow('canvas',output_canvas)
            cv2.waitKey(0) 
            decision_made=True
    else:
        print("line is not visible")


##################################################### GMS ###############################################################################################