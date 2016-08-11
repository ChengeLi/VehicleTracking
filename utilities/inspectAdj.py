

"""inspect the adj elements for the two vehicles"""
# FeatureMtxLoc = pickle.load(open(os.path.join(DataPathobj.adjpath,'veryZigWhiteCar_FeatureMtxLoc'),'rb'))
# vehicle1ind = 0
# vehicle2ind = 1
# """adj element between two vehicles:"""
# print np.unique((fully_adj+fully_adj.T)[np.array(FeatureMtxLoc[vehicle1ind]),:][:,np.array(FeatureMtxLoc[vehicle2ind])])

# print np.unique((fully_adj+fully_adj.T)[np.array(FeatureMtxLoc[vehicle1ind]),:][:,np.array(FeatureMtxLoc[vehicle1ind])])
# print np.unique((fully_adj+fully_adj.T)[np.array(FeatureMtxLoc[vehicle2ind]),:][:,np.array(FeatureMtxLoc[vehicle2ind])])



        
def diff_feature_on_one_car(dataForKernel,feature_diff_tensor, trjID):
    """inspect features for one car or two cars"""
    # one_car_trjID = pickle.load(open('./johnson_one_car_trjID','rb'))
    # one_car_trjID = pickle.load(open(os.path.join(DataPathobj.adjpath,'one_car_trjID'),'rb'))
    # one_car_trjID = pickle.load(open(os.path.join(DataPathobj.adjpath,'veryZigWhiteCar'),'rb'))

    bus_and_car_trjID = [[0,3,204,764,1428,1218],[1092,138,921,934,54,65]]
    one_car_trjID = bus_and_car_trjID

    FeatureMtxLoc    = {}
    dist_one_car     = {}
    vxdist_one_car   = {}
    vydist_one_car   = {}
    blobdist_one_car = {}
    for ii in range(np.array(one_car_trjID).shape[0]):
        FeatureMtxLoc[ii]    = []
        dist_one_car[ii]     = []
        vxdist_one_car[ii]   = []
        vydist_one_car[ii]   = []
        blobdist_one_car[ii] = []
        feature_diff_tensor2 = feature_diff_tensor
        feature_diff_tensor2[np.isnan(feature_diff_tensor2)]=0
        for aa in np.array(one_car_trjID)[ii]:
            FeatureMtxLoc[ii]+=list(np.where(trjID==aa)[0])
            dist_one_car[ii]     = np.max((feature_diff_tensor2[:,:,2]+feature_diff_tensor2[:,:,2].T)[np.array(FeatureMtxLoc[ii]),:][:,np.array(FeatureMtxLoc[ii])])
            vxdist_one_car[ii]   = np.max((feature_diff_tensor2[:,:,0]+feature_diff_tensor2[:,:,0].T)[np.array(FeatureMtxLoc[ii]),:][:,np.array(FeatureMtxLoc[ii])])
            vydist_one_car[ii]   = np.max((feature_diff_tensor2[:,:,1]+feature_diff_tensor2[:,:,1].T)[np.array(FeatureMtxLoc[ii]),:][:,np.array(FeatureMtxLoc[ii])])
            blobdist_one_car[ii] = np.max((feature_diff_tensor2[:,:,4]+feature_diff_tensor2[:,:,4].T)[np.array(FeatureMtxLoc[ii]),:][:,np.array(FeatureMtxLoc[ii])])
    
    # interesting_adj_part     = (adj[np.array(FeatureMtxLoc[ii]),:][:,np.array(FeatureMtxLoc[ii])]>0).astype(int)
    # interesting_fulladj_part = ((fully_adj+fully_adj.T)[np.array(FeatureMtxLoc[ii]),:][:,np.array(FeatureMtxLoc[ii])]>0).astype(int)
    # pickle.dump(FeatureMtxLoc,open(os.path.join(DataPathobj.adjpath,'veryZigWhiteCar_FeatureMtxLoc'),'wb'))
    np.max(dist_one_car.values())
    np.max(vxdist_one_car.values())
    np.max(vydist_one_car.values())
    np.max(blobdist_one_car.values())

    feature_dim = 2
    vehicle1ind = 0
    vehicle2ind = 1
    """distance between two vehicles:"""
    print np.unique((feature_diff_tensor[:,:,feature_dim]+feature_diff_tensor[:,:,feature_dim].T)\
        [np.array(FeatureMtxLoc[vehicle1ind]),:][:,np.array(FeatureMtxLoc[vehicle2ind])])

    """distance within one vehicle:"""
    print np.unique((feature_diff_tensor[:,:,feature_dim]+feature_diff_tensor[:,:,feature_dim].T)\
        [np.array(FeatureMtxLoc[vehicle1ind]),:][:,np.array(FeatureMtxLoc[vehicle1ind])])

    print np.unique((feature_diff_tensor[:,:,feature_dim]+feature_diff_tensor[:,:,feature_dim].T)\
        [np.array(FeatureMtxLoc[vehicle2ind]),:][:,np.array(FeatureMtxLoc[vehicle2ind])])

    pdb.set_trace()

def adj_GTind(fully_adj,trjID):
    """use fake GT ind to group?"""
    rearrange_adj = np.zeros_like(fully_adj)
    matidx = 0
    global_annolist = pickle.load(open(DataPathobj.DataPath+'/'+str(matidx).zfill(3)+'global_annolist.p','rb'))
    _, idx = np.unique(global_annolist, return_index=True)
    unique_anno = np.array(global_annolist)[np.sort(idx)]

    arrange_index = []
    for ii in range(fully_adj.shape[0]):
        arrange_index = arrange_index+ list(np.where(trjID==unique_anno[ii])[0])

    rearrange_adj = fully_adj[arrange_index,:][:,arrange_index]
    plt.figure()
    plt.imshow(rearrange_adj)
    pdb.set_trace()
    pickle.dump(arrange_index,open(DataPathobj.DataPath+'/arrange_index.p','wb'))
    return rearrange_adj



def CCoverseg(matidx,adj,fully_adj,sameDirTrjID,x,y,non_isolatedCC):
    # if matidx==1:
    #     big_CC_trjID = pickle.load(open(os.path.join(DataPathobj.DataPath,'NGSIM_bigCC_trjID_2ndTrunc'),'rb'))
    # if matidx==2:
    #     big_CC_trjID = pickle.load(open(os.path.join(DataPathobj.DataPath,'NGSIM_bigCC_trjID_3rdTrunc'),'rb'))

    # CC_overseg_trjID = pickle.load(open(os.path.join(DataPathobj.DataPath,'CC_overseg_trjID'),'rb'))
    # interesting_trjIDdic = big_CC_trjID
    # interesting_trjIDdic = CC_overseg_trjID


    # interesting_trjID = [ 921, 1754, 1903, 2032, 2120, 2325, 2584]
    # interesting_trjID =  [  77,  104,  295,  330,  367,  445,  518,  606,  723,  840,  855,
    #     865,  891, 1138, 1362, 1724, 1745]

    interesting_trjID = [2887, 2896, 3000, 3399, 3609,       2714, 2735, 2755, 2764, 2844, 2976, 3192, 4004]
    FeatureMtxLoc_CC = {}
    """interested in connected components, every CC has many trj IDs"""
    # for key in interesting_trjIDdic.keys():
    #     FeatureMtxLoc_CC [key] = []
    #     for aa in interesting_trjIDdic[key]:
    #         # print list(np.where(sameDirTrjID==aa)[0])
    #         FeatureMtxLoc_CC[key]+= list(np.where(sameDirTrjID==aa)[0])

    """interested in trj directly"""
    for key in interesting_trjID:
        location = list(np.where(sameDirTrjID==key)[0])
        if location:
            FeatureMtxLoc_CC [key] = []
            FeatureMtxLoc_CC[key]+= location


    np.where(c==56)[0]


    """distance within one connected component:"""
    # for key in interesting_trjIDdic.keys():
    #     interesting_adj_part = (adj[np.array(FeatureMtxLoc_CC[key]),:][:,np.array(FeatureMtxLoc_CC[key])]>0).astype(int)
    #     interesting_fulladj_part = ((fully_adj+fully_adj.T)[np.array(FeatureMtxLoc_CC[key]),:][:,np.array(FeatureMtxLoc_CC[key])]>0).astype(int)
    #     temp_dist = feature_diff_tensor[:,:,2]
    #     temp_dist[np.isnan(temp_dist)] = 0
    #     dist_these_CCs = temp_dist[np.array(FeatureMtxLoc_CC[key]),:][:,np.array(FeatureMtxLoc_CC[key])].astype(int)
    #     x_these_CCs = x[np.array(FeatureMtxLoc_CC[key]),:]
    #     y_these_CCs = y[np.array(FeatureMtxLoc_CC[key]),:]

    """interested in trj directly"""
    interesting_loc = np.reshape(np.array(FeatureMtxLoc_CC.values()),(-1,))
    interesting_adj_part = (adj[interesting_loc,:][:,interesting_loc]>0).astype(int)
    interesting_fulladj_part = (fully_adj[interesting_loc,:][:,interesting_loc]>0).astype(int)
    
    normalized_feature_diff_tensor2 = normalized_feature_diff_tensor[non_isolatedCC,:][:,non_isolatedCC] ## use the newly normalized tensor
    pdb.set_trace()
    normalized_feature_diff_tensor2[np.isnan(normalized_feature_diff_tensor2)] = 0
    temp_dist = normalized_feature_diff_tensor2[:,:,2]+normalized_feature_diff_tensor2[:,:,2].T
    dist_these_CCs = temp_dist[interesting_loc,:][:,interesting_loc].astype(int)
    Vx_dist_these_CCs = (normalized_feature_diff_tensor2[:,:,0]+normalized_feature_diff_tensor2[:,:,0].T)[interesting_loc,:][:,interesting_loc].astype(int)
    Vy_dist_these_CCs = (normalized_feature_diff_tensor2[:,:,1]+normalized_feature_diff_tensor2[:,:,1].T)[interesting_loc,:][:,interesting_loc].astype(int)

    x_these_CCs = x[non_isolatedCC,:][interesting_loc,:]
    y_these_CCs = y[non_isolatedCC,:][interesting_loc,:]


    CClabelforinterestingTrjID = c[interesting_loc]

    if np.sum(interesting_adj_part!= interesting_fulladj_part*(Vx_dist_these_CCs< 1))>1:
        pdb.set_trace()


    # plt.figure()
    # for hh in range(len(FeatureMtxLoc_CC[key])):
    #     featureloc = FeatureMtxLoc_CC[key][hh]
    #     plt.scatter(x[non_isolatedCC,:][featureloc,:],y[non_isolatedCC,:][featureloc,:])
    #     plt.draw()
    #     plt.show()
    plt.figure()
    for hh in range(len(interesting_loc)):
        plt.scatter(x[non_isolatedCC,:][interesting_loc,:],y[non_isolatedCC,:][interesting_loc,:])
        plt.draw()
        plt.show()

    pdb.set_trace()


