
# convolve with 1/N
# moving average
x = np.linspace(0,2*np.pi,100)
y = np.sin(x) + np.random.random(100) * 0.8

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
plt.figure()
plt.plot(x, y,'o')
plt.plot(x, smooth(y,3), 'r-', lw=2)
plt.plot(x, smooth(y,9), 'g-', lw=2)
plt.plot(x, smooth(y,12), 'y-', lw=2)






import statsmodels.api as sm
## fit (x,y)
lowessXY = sm.nonparametric.lowess(yk, xk, frac=0.1)
plt.figure()
plt.plot(xk, yk, '+')
plt.plot(lowessXY[:, 0], lowessXY[:, 1])
plt.show()


#fit x(t) and y(t) seperately
lowessX = sm.nonparametric.lowess(xk,range(len(xk)), frac=0.1)
plt.figure()
plt.plot(range(len(xk)), xk, '+')
plt.plot(lowessX[:, 0], lowessX[:, 1])
plt.show()


lowessY = sm.nonparametric.lowess(yk,range(len(yk)), frac=0.1)
plt.figure()
plt.plot(range(len(yk)), yk, '+')
plt.plot(lowessY[:, 0], lowessY[:, 1])
plt.show()
























