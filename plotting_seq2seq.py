
import matplotlib.pyplot as plt
import numpy as np

def get_validation(data, searched_string):
    res = []
    for sub in data:
        if searched_string in sub:
            res.append(float(sub.split(searched_string+": ")[1]))
            #print(sub)
    return res


if __name__ == "__main__":
    x = []
    x2 = []
    x3 = []
    x4 = []
    path = "C:/Users/Meier/Projekte/Plots/europarl/" # provide path to training output
    extraction_target = "perplexity"
    with open(path+"baseline_europarl.txt", "r", encoding = "utf-8") as f:
        for line in f:
            if "INFO" in line:
                x.extend(line.split("\n"))
    x = get_validation(x, extraction_target)
    with open(path+"dynamic_europarl.txt", "r", encoding = "utf-8") as f:
        for line in f:
            if "INFO" in line:
                x2.extend(line.split("\n"))
    x2 = get_validation(x2, extraction_target)
    with open(path + "cf_dnn2_deeper_constant_lr.txt", "r", encoding = "utf-8") as f:
        for line in f:
            if "INFO" in line:
                x3.extend(line.split("\n"))
    x3 = get_validation(x3, extraction_target)
    with open(path + "cf_var1_dnn2_cl.txt", "r", encoding = "utf-8") as f:
        for line in f:
            if "INFO" in line:
                x4.extend(line.split("\n"))
    x4 = get_validation(x4, extraction_target)
    print("x2: ", x2)
    
    #"""
    epochs = [i for i in range(10000,110000) if i%10000 == 0]
    #val_scores = [56.42, 59.33, 60.43, 61.05, 61.30, 62.70, 63.26, 63.41, 63.53, 63.56]
    #val_perp = [9.51, 7.36, 6.79, 6.58, 6.53, 6.04, 5.84, 5.81, 5.89, 5.91]
    xent = []
    #plt.plot
    #plt.plot(epochs, val_perp)
    x = x[:10]
    x = np.array(x)
    x2 = x2[:10]
    x2 = np.array(x2)
    x3 = x3[:10]
    x3 = np.array(x3)
    x4 = x4[:10]
    x4 = np.array(x4)
    print(x)
    epochs = np.array(epochs)
    plt.plot(epochs, x, "r", label = "Baseline")
    plt.plot(epochs,x2, "b", label = "Dynamic Fusion")
    plt.plot(epochs,x3, "g", label = "Cold Fusion")
    plt.plot(epochs,x4, "y", label = "CF Variant 1")
    
    plt.xlabel("Epochs")
    plt.ylabel("Validation Perplexity")
    plt.legend(loc="upper right")
    plt.show()
    print(epochs) 
    #"""
