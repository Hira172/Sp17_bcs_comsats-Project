from math import sqrt

def time_to_contact(original_frame,matchedBoxes, newTime, oldTime, key_list, val_list):
    tracked_bboxes = []
    if matchedBoxes == []:
        return tracked_bboxes

    dt = newTime - oldTime
    print("DT:\t\t",dt)
    data = matchedBoxes

    for track, detect in data:
        tbbox = track.to_tlbr()
        dbbox = detect.to_tlbr()
        class_name = track.get_class()  # Get the class name of particular object
        tracking_id = track.track_id # Get the ID for the particular track
        index = key_list[val_list.index(class_name)] # Get predicted object index by object name
        S = dbbox[2] / tbbox[2]
        if S > 1:
            print("S\t\t",S)
            tm = dt / (S-1)
            print("S-1\t\t",S-1)
            print("tm\t\t", tm)
            C = tm + 1
            print("C\t\t",C)

            sqrt_val = 1+(2*C)
            print("sqrt\t\t",sqrt_val)
            num = 1-sqrt(sqrt_val)
            print("NUME\t\t",num)
            T = (num/C)*tm
            print("T\t\t",T)
            print("\n\n\n\n")
            # adding the tracking id , index and time to collusion to the bbox np.array
            tracked_bboxes.append(dbbox.tolist() + [tracking_id, index, tm])  # Structure data, that we could use it with our draw_bbox function
        else:
            tracked_bboxes.append(dbbox.tolist() + [tracking_id, index])

    return tracked_bboxes
