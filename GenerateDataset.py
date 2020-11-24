import matplotlib.pyplot as plt
import numpy as np
import math
import csv
import sys
import os
import scipy.stats
import dask.dataframe
#from __builtin__ import True


synthDatasetname= 'synth2Ddense'

#the parameters of the mean lines. a is the slope, c is the starting point at x=0
p_a=9.0/6.0 #height/width
p_c=-p_a*4
n_a=6.0/9.0
n_c=4
b=-1

#the gaussian distribution horizontally
p_h_mu=17.3
p_h_sigma=5
n_h_mu=10
n_h_sigma=5

#the gaussian distribution vertically
p_v_mu=0
p_v_sigma=0.8
n_v_mu=0
n_v_sigma=0.8

#number of pos and neg points
p_n=2000
n_n=8000

#the cut-off boundaries
min_x=0
max_x=10
min_y=0
max_y=10

numRules=-1
filename=synthDatasetname+"Plot.pdf"

seed=1

resolution_grid=101



# def prob(x,y):
#     #this is not exactly the probability, will perhaps change it to cdf or 10/1000
#     prob_p=scipy.stats.norm(mu, variance).pdf(distanceFromLine(x, y, p_a, b, p_c))
#     prob_n=scipy.stats.norm(mu, variance).pdf(distanceFromLine(x, y, n_a, b, n_c))
#     prob_xy=

#https://matplotlib.org/3.1.0/gallery/userdemo/colormap_normalizations_custom.html
import matplotlib.colors as colors
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def distanceFromLine(x,y,a,b,c):
    #from https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    return np.abs(a*x+b*y+c)/np.sqrt(a*a+b*b)

def projectionOnLine(x,y,a,b,c):
    #from https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    return ((b(b*x-a*y)-ac)/(a*a+b*b),(a(-b*x+a*y)-bc)/(a*a+b*b))

def generatePointsAroundLineNonRotated(numSamples,a,c,h_mu,h_sigma,v_mu,v_sigma):
    x=np.random.normal(h_mu,h_sigma,numSamples*10)
    x=np.extract(x<=max_x,x)
    x=np.extract(x>=min_x,x)
    x=x[0:numSamples]
    #print x
    y=np.random.normal(v_mu,v_sigma,numSamples)

    #just add up the function
    y=c+a*x + y

    #rotation actually looks weard, just add up the function
    #y=y+p_c
    #rotation=np.array([[1,p_a],[-p_a,1]])
    #xy=np.array([x,y]).T
    #xy=np.matmul(xy,rotation)
    #print xy
    #plt.scatter(xy[:,0],xy[:,1])
    return x,y

def generatePointsAroundLine(numSamples,a,c,h_mu,h_sigma,v_mu,v_sigma):
    x=np.random.normal(h_mu,h_sigma,numSamples*100)
    #print x
    y=np.random.normal(v_mu,v_sigma,numSamples*100)

    #just add up the function
    #y=c+a*x + y

    #y=y+c

    angle= math.atan(-a)#
    rotation=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
#    xy=np.array([x,y]).T
    xy=np.column_stack((x,y))
    xy=np.matmul(xy,rotation)
    xy[:,1]=xy[:,1]+c
    #print xy
    #plt.scatter(xy[:,0],xy[:,1])
    #todo: adapt
    #x=np.extract(x<=max_x,x)
    #x=np.extract(x>=min_x,x)
    xy= xy[(xy[:,0]>=min_x) & (xy[:,0]<=max_x) & (xy[:,1]<=max_y) & (xy[:,1]>=min_y)]
    #xy=np.extract(xy[0]<=max_x,x)
    xy=xy[0:numSamples]
    return xy[:,0],xy[:,1]


def writeDataset(x_p,y_p,x_n,y_n,filename,relationname):
    f = open(filename, "w")
    f.write("@relation %s\n"%(relationname))
    f.write("@attribute x numeric\n")
    f.write("@attribute y numeric\n")
    f.write("@attribute class {1,0}\n")
    f.write("@data\n")
    for x,y in zip(x_p,y_p):
        f.write("%8.8f,%8.8f,%s\n"%(x,y,'1'))
    for x,y in zip(x_n,y_n):
        f.write("%8.8f,%8.8f,%s\n"%(x,y,'0'))
    f.close()

def writeDatasetLinear(xx,yy,zz,filename,relationname):
    f = open(filename, "w")
    f.write("@relation %s\n"%(relationname))
    f.write("@attribute x numeric\n")
    f.write("@attribute y numeric\n")
    f.write("@attribute class {1,0}\n")
    f.write("@data\n")
    for x,y,z in zip(xx,yy,zz):
        f.write("%8.8f,%8.8f,%s\n"%(x,y,z))
    f.close()


def fastloadCSV(filename):
    #TODO: probably faster options: https://medium.com/casual-inference/the-most-time-efficient-ways-to-import-csv-data-in-python-cc159b44063d
#    csvReader = csv.reader(open(filename), delimiter=",")
#    data=list(csvReader)
#    data=np.array(data).astype("integer")

    #still too much memory
    #data=np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=0)

    with open(filename) as f:
        no_columns=len(list(f.readline().split(",")))

    no_rows=0
    with open(filename) as f:
        for i, l in enumerate(f):
            pass
        no_rows=i+1

    
#    print("identified %d columns and %d rows"%(no_columns,no_rows)

#    data = np.zeros((no_rows, no_columns), dtype = np.integer)
    data = np.zeros((no_rows, no_columns), dtype = np.float16)
#    print "created numpy matrix"

    with open(filename) as f:
        for i, line in enumerate(f):
            data[i,:] = np.array(line.split(","))
#    print "loaded data into numpy array"


    return data


if __name__ == '__main__':


#    print "test"



    np.random.seed(seed)
    plt.figure(figsize=(5,5))
    x = np.linspace(min_x, max_x, 1000)
    plt.plot(x, p_c+p_a*x, color='blue');
    plt.plot(x, n_c+n_a*x, color='red');

    x_p,y_p=generatePointsAroundLine(p_n, p_a, p_c,p_h_mu,p_h_sigma,p_v_mu,p_v_sigma)
    x_n,y_n=generatePointsAroundLine(n_n, n_a, n_c,n_h_mu,n_h_sigma,n_v_mu,n_v_sigma)


    writeDataset(x_p,y_p,x_n,y_n,synthDatasetname+"_train.arff",synthDatasetname+"-train_p-%d_n-%d_seed-%d"%(p_n,n_n,seed))

    seed=seed+1
    x_p_t,y_p_t=generatePointsAroundLine(p_n, p_a, p_c,p_h_mu,p_h_sigma,p_v_mu,p_v_sigma)
    x_n_t,y_n_t=generatePointsAroundLine(n_n, n_a, n_c,n_h_mu,n_h_sigma,n_v_mu,n_v_sigma)
    writeDataset(x_p_t,y_p_t,x_n_t,y_n_t,synthDatasetname+"_test.arff",synthDatasetname+"-test_p-%d_n-%d_seed-%d"%(p_n,n_n,seed))


    #first, just plotting (smoother)
    grid = np.linspace(min_x, max_x, 1000)
    xx,yy = np.meshgrid(grid,grid)
    zz= (distanceFromLine(xx,yy,p_a,b,p_c)<distanceFromLine(xx,yy,n_a,b,n_c)).astype(int)

    if(False):
        plt.contour(grid,grid,zz,colors='green',levels=[0,1],linestyles='dashed' )
    if(True):
        plt.contour(grid,grid,zz,colors='gray',levels=[0,1],linestyles='dashed' )

    #then, generate grid data set
    grid = np.linspace(min_x, max_x, resolution_grid)
    xx,yy = np.meshgrid(grid,grid)
    zz= (distanceFromLine(xx,yy,p_a,b,p_c)<distanceFromLine(xx,yy,n_a,b,n_c)).astype(int)


    xx= xx.reshape(-1)
    yy= yy.reshape(-1)
    zz= zz.reshape(-1)

    #print x_p
    writeDatasetLinear(xx,yy,zz,synthDatasetname+"_grid.arff",synthDatasetname+"-test_p-%d_n-%d_seed-%d"%(resolution_grid*resolution_grid/2,resolution_grid*resolution_grid/2,seed))

    #now, plot the rules
    #print xx.shape
    #print xx.reshape(resolution_grid,resolution_grid).shape
    #plt.title("synthetic data")

    if len(sys.argv)>2:
        numRules=int(sys.argv[2])
#    print "number of rules to show",numRules

    if numRules>0:

    #    filename=sys.argv[1]
        filename=sys.argv[1]
        data=fastloadCSV(filename)
        if numRules>data.shape[1]-4:
            numRules=data.shape[1]-4

#        print "plot %s rules"%numRules
    #     print ("loading")
    #     data = dask.dataframe.read_csv(filename)
    #     #data.set_index(0)
    #     print ("loaded")
    #     print data.columns
    #     print data.loc[:,1]

        res=int(math.sqrt(data.shape[0]))

        xx=data[:,1].reshape(res,res)
        #xx=xx.reshape(resolution_grid,resolution_grid)
        yy=data[:,2].reshape(res,res)
        #yy=yy.reshape(resolution_grid,resolution_grid)

        #for i in [1,5,10,15,30]:
        coverage=np.zeros(xx.shape)


        for i in xrange(numRules):
            zz=data[:,i+4].reshape(res,res)
            coverage=coverage+zz
#            print i, sum(zz.reshape(-1)),sum(coverage.reshape(-1)), xx[0,0],yy[0,0],zz[0,0] 
            #plt.contour(xx,yy,zz,colors='gray',levels=1,linestyles='solid' )
            if sum(zz.reshape(-1))>0:
                plt.contour(xx,yy,zz,colors='blue',levels=1,linestyles='dashed',linewidths=1.5 )
            else:
                plt.contour(xx,yy,zz,colors='red',levels=1,linestyles='dashed',linewidths=1.5 )

        min_level=int(min(coverage.reshape(-1)))
        max_level=int(max(coverage.reshape(-1)))
        level=max(-min_level,max_level)
#        print min_level,max_level,level
    #    print min_level,max_level,level
    #    plt.contourf(grid,grid,zz)
    #ValueError: Colormap RdGr is not recognized. Possible values are: Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Vega10, Vega10_r, Vega20, Vega20_r, Vega20b, Vega20b_r, Vega20c, Vega20c_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, inferno_r, jet, jet_r, magma, magma_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, seismic, seismic_r, spectral, spectral_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, viridis, viridis_r, winter, winter_r
        #plt.contourf(xx,yy,coverage,levels=range(-20,4),cmap='RdYlGn')
    #    plt.contourf(xx,yy,coverage,cmap='RdYlGn', alpha=1.0,norm=MidpointNormalize(vmin=-20,vmax=10,midpoint=0.))
#        plt.contourf(xx,yy,coverage,cmap='RdYlBu', alpha=1.0,levels=range(min_level,max_level),norm=MidpointNormalize(vmin=-level,vmax=level,midpoint=0.))
        plt.contourf(xx,yy,coverage,30,cmap='RdYlGn',alpha=0.6)
        plt.colorbar()
#        print zip(xx,yy,zz)
        if(False):
            plt.title(os.path.basename(filename)+" 1 to "+str(numRules)+" rules")
        #plot decision boundary
        plt.contour(xx,yy,coverage,levels=[0],colors='green',linestyles='solid',linewidths=2)
        data=None

    #plt.scatter(x_n,y_n,color='red')
    #plt.scatter(x_p,y_p,color='blue')

    #verhindert etwas das ueberlappen einer punktmenge der anderen
    for i in range(0,x_n.shape[0],50):
        plt.scatter(x_n[i:i+50],y_n[i:i+50],color='red',s=0.05)
        plt.scatter(x_p[i:i+50],y_p[i:i+50],color='blue',s=0.05)


#    plt.scatter(x_p_t,y_p_t)
#    plt.scatter(x_n_t,y_n_t)


    plt.xlim(0,10)
    plt.ylim(0,10)
    plt.tight_layout()
    if True:
        plt.show()
    else:
        outFile=os.path.basename(filename).replace('.csv',"_rules-%d.pdf"%numRules)
        plt.savefig(outFile, bbox_inches='tight', pad_inches=0)
#        print "written to",outFile
    plt.clf()
    plt.close()

