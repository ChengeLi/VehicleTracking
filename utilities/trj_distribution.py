# This module is used to inspect the distributions of the trjactories 
# in order to better understand, and see the result after projective mapping

matfiles = sorted('../DoT/CanalSt@BaxterSt-96.106/adj/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/trj_*.mat'))
lrsl = '../DoT/CanalSt@BaxterSt-96.106/finalresult/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/result' 



def diff_distribution():
    n, bins, patches = plt.hist(sxdiffAll, 100, normed=1, facecolor='green', alpha=0.75)
    n, bins, patches = plt.hist(sydiffAll, 100, normed=1, facecolor='green', alpha=0.75)
    n, bins, patches = plt.hist(mdisAll, 100, normed=1, facecolor='green', alpha=0.75)
    n, bins, patches = plt.hist(centerDisAll, 100, normed=1, facecolor='green', alpha=0.75)
    n, bins, patches = plt.hist(huedisAll, 100, normed=1, facecolor='green', alpha=0.75)
    plt.draw()
    plt.show()

    pickle.dump(sxdiffAll,open('./sxdiffAll_johnson','wb'))
    pickle.dump(sydiffAll,open('./sydiffAll_johnson','wb'))
    pickle.dump(mdisAll,open('./mdisAll_johnson','wb'))
    pickle.dump(centerDisAll, open('./centerDisAll_johnson','wb'))
    pickle.dump(huedisAll,open('./huedisAll_johnson','wb'))























