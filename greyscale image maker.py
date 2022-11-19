from PIL import Image
import os

def exists(path):
    """Test whether a path exists.  Returns False for broken symbolic links"""
    try:
        st = os.stat(path)
    except os.error:
        return False
    return True


"""USER INPUTS"""
particles = ['ele','muo','pro','pio','kao']
pwd = 'D:\\uni\\Y4\\Project\\Particle_csv_files-20171122T194152Z-001\\Particle_csv_files\\'
pwd2 = 'graphs\\Colour Image Set\\'
planes = [0,1,2]

evtnum_start = 1                    #USER INPUT
evtnum_end = 1000                   #USER INPUT


prob_train = 0.3
"""STARTS HERE"""



for ele in particles:
    for plane in planes:
        evts = []
        for evt in range(evtnum_start, evtnum_end + 1):
            test_dir = pwd + pwd2 + str(ele) + '_' + str(evt) + '_' + str(plane) + '.png'
            if exists(test_dir) == True:
                evts.append(evt)
        for evt in evts:
            if (evt%100 == 0):
                print("Running %s \t plane %d \t event %d" % (ele, plane,evt))
            img_dir = pwd2 + str(ele) + '_' + str(evt) + '_' + str(plane) + '.png'
            img = Image.open(img_dir).convert('L')
            new_dir = 'graphs\\Greyscale Image Set\\' + str(ele) + '_' + str(evt) + '_' + str(plane) + '.png'
            img.save(new_dir)

#img = Image.open('image.png.).con

