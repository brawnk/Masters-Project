from PIL import Image
import os


if __name__ == "__main__":
    particles = ['ele', 'muo', 'pro', 'pio', 'kao']
    pwd = 'D:\\uni\\Y4\\Project\\Particle_csv_files-20171122T194152Z-001\\Particle_csv_files\\'
    pwd2 = 'graphs\\Colour Image Set\\'
    planes = [0, 1, 2]

    evtnum_start = 1
    evtnum_end = 1000

    for ele in particles:
        for plane in planes:
            evts = []
            for evt in range(evtnum_start, evtnum_end + 1):
                test_dir = pwd + pwd2 + str(ele) + '_' + str(evt) + '_' + str(plane) + '.png'
                if os.path.exists(test_dir):
                    evts.append(evt)
            for evt in evts:
                if evt % 100 == 0:
                    print("Running %s \t plane %d \t event %d" % (ele, plane, evt))
                img_dir = pwd2 + str(ele) + '_' + str(evt) + '_' + str(plane) + '.png'
                img = Image.open(img_dir).convert('L')
                new_dir = 'graphs\\Greyscale Image Set\\' + str(ele) + '_' + str(evt) + '_' + str(plane) + '.png'
                img.save(new_dir)

    # img = Image.open('image.png.).con
