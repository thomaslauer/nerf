import matplotlib.pyplot as plt

def makeHist(image, bins=50):

    print("Hello world")

    r = image[:,:,0].flatten()
    g = image[:,:,1].flatten()
    b = image[:,:,2].flatten()

    plt.cla()

    plt.subplot(2,2,1)
    plt.imshow(image)

    plt.subplot(2,2,2)
    plt.title("R")
    plt.hist(r, bins=bins, color='red')

    plt.subplot(2,2,3)
    plt.title("G")
    plt.hist(g, bins=bins, color='green')

    plt.subplot(2,2,4)
    plt.title("B")
    plt.hist(b, bins=bins, color='blue')

    plt.show()


