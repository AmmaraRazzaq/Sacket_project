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


