
import numpy as np
import matplotlib.pyplot as plt



#exec(open("final_code_streamfunction2.py").read())
#exec(open("new2_single_11.py").read())


#exec(open("new2_single_12.py").read())

#exec(open("stream_function.py").read())

exec(open("new2_single_BNN_final_IG.py").read())
A_IG=A_mean_baseline1_24_1
A_IG_std=stds_24_1 
A_IG2=A_mean_baseline1_96_1
A_IG_std2=stds_96_1
exec(open("new2_single_BNN_final_SHAP.py").read())
A_SHAP=A_mean_baseline1_24_1
A_SHAP_std=stds_24_1
A_SHAP2=A_mean_baseline1_96_1
A_SHAP_std2=stds_96_1
exec(open("new2_single_BNN_final_deeplift.py").read())
A_deeplift=A_mean_baseline1_24_1
A_deeplift_std=stds_24_1
A_deeplift2=A_mean_baseline1_96_1
A_deeplift_std2=stds_96_1
print(A_IG.shape)
years=np.arange(1993,2015,1/12)
for w in range(1):
    w=0
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(years,np.mean(A_IG[:,:,0],axis=1)*100,'r')
    #shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
    plt.ylabel('IG_PT*100',fontsize=11)
    # Set y-axis to log scale
    #plt.xlabel('Input feature',fontsize=13)
    #plt.ylabel('feature',fontsize=13)
    #plt.xlabel('composite_shap',fontsize=13)
    plt.subplot(3,1,2)
    plt.plot(years,np.mean(A_SHAP[:,:,0],axis=1)*100,'g')
    #shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
    plt.ylabel('SHAP_PT*100',fontsize=11)
    # Set y-axis to log scale
    #plt.xlabel('Time',fontsize=13)
    #plt.ylabel('feature',fontsize=13)
    #plt.xlabel('composite_shap',fontsize=13)
    plt.subplot(3,1,3)
    plt.plot(years,np.mean(A_deeplift[:,:,0],axis=1)*100,'b')
    #shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
    plt.ylabel('DL_PT*100',fontsize=11)
    # Set y-axis to log scale
    plt.xlabel('Time',fontsize=11)
    plt.savefig("ig_composite_2_NADW_time_total_"+str(w)+".png")

for w in range(1):
    w=1

    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(years,np.mean(A_IG[:,:,1],axis=1)*100,'r')
    #shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
    plt.ylabel('IG_Salinity*100',fontsize=11)
    # Set y-axis to log scale
    #plt.xlabel('Input feature',fontsize=13)
    #plt.ylabel('feature',fontsize=13)
    #plt.xlabel('composite_shap',fontsize=13)
    plt.subplot(3,1,2)
    plt.plot(years,np.mean(A_SHAP[:,:,1],axis=1)*100,'g')
    #shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
    plt.ylabel('SHAP_Salinity*100',fontsize=11)
    # Set y-axis to log scale
    #plt.xlabel('Time',fontsize=13)
    #plt.ylabel('feature',fontsize=13)
    #plt.xlabel('composite_shap',fontsize=13)
    plt.subplot(3,1,3)
    plt.plot(years,np.mean(A_deeplift[:,:,1],axis=1)*100,'b')
    #shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
    plt.ylabel('DL_Salinity*100',fontsize=11)
    # Set y-axis to log scale
    plt.xlabel('Time',fontsize=11)
    plt.savefig("ig_composite_2_NADW_time_total_"+str(w)+".png")

for w in range(1): 
    w=2
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(years,np.mean(A_IG[:,:,2],axis=1)*100,'r')
    #shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
    plt.ylabel('IG_EKE*100',fontsize=11)
    # Set y-axis to log scale
    #plt.xlabel('Input feature',fontsize=13)
    #plt.ylabel('feature',fontsize=13)
    #plt.xlabel('composite_shap',fontsize=13)
    plt.subplot(3,1,2)   
    plt.plot(years,np.mean(A_SHAP[:,:,2],axis=1)*100,'g')
    #shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
    plt.ylabel('SHAP_EKE*100',fontsize=11)
    # Set y-axis to log scale
    #plt.xlabel('Time',fontsize=13)
    #plt.ylabel('feature',fontsize=13)
    #plt.xlabel('composite_shap',fontsize=13)
    plt.subplot(3,1,3)
    plt.plot(years,np.mean(A_deeplift[:,:,2],axis=1)*100,'b')
    #shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
    plt.ylabel('DL_EKE*100',fontsize=11)
    # Set y-axis to log scale
    plt.xlabel('Time',fontsize=11)
    plt.savefig("ig_composite_2_NADW_time_total_"+str(w)+".png")


for w in range(1):
    w=3
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(years,np.mean(A_IG[:,:,3],axis=1)*100,'r')
    #shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
    plt.ylabel('IG_N*100',fontsize=11)
    # Set y-axis to log scale
    #plt.xlabel('Input feature',fontsize=13)
    #plt.ylabel('feature',fontsize=13)
    #plt.xlabel('composite_shap',fontsize=13)
    plt.subplot(3,1,2)
    plt.plot(years,np.mean(A_SHAP[:,:,3],axis=1)*100,'g')
    #shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
    plt.ylabel('SHAP_N*100',fontsize=11)
    # Set y-axis to log scale
    #plt.xlabel('Time',fontsize=13)
    #plt.ylabel('feature',fontsize=13)
    #plt.xlabel('composite_shap',fontsize=13)
    plt.subplot(3,1,3)
    plt.plot(years,np.mean(A_deeplift[:,:,3],axis=1)*100,'b')
    #shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
    plt.ylabel('DL_N*100',fontsize=11)
    # Set y-axis to log scale
    plt.xlabel('Time',fontsize=11)
    plt.savefig("ig_composite_2_NADW_time_total_"+str(w)+".png")


for w in range(1):
    w=0
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(years,np.mean(A_IG2[:,:,0],axis=1)*100,'r')
    #shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
    plt.ylabel('IG_PT*100',fontsize=11)
    # Set y-axis to log scale
    #plt.xlabel('Input feature',fontsize=13)
    #plt.ylabel('feature',fontsize=13)
    #plt.xlabel('composite_shap',fontsize=13)
    plt.subplot(3,1,2)
    plt.plot(years,np.mean(A_SHAP2[:,:,0],axis=1)*100,'g')
    #shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
    plt.ylabel('SHAP_PT*100',fontsize=11)
    # Set y-axis to log scale
    #plt.xlabel('Time',fontsize=13)
    #plt.ylabel('feature',fontsize=13)
    #plt.xlabel('composite_shap',fontsize=13)
    plt.subplot(3,1,3)
    plt.plot(years,np.mean(A_deeplift2[:,:,0],axis=1)*100,'b')
    #shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
    plt.ylabel('DL_PT*100',fontsize=11)
    # Set y-axis to log scale
    plt.xlabel('Time',fontsize=11)
    plt.savefig("ig_composite_8_NADW_time_total_"+str(w)+".png")

for w in range(1):
    w=1

    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(years,np.mean(A_IG2[:,:,1],axis=1)*100,'r')
    #shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
    plt.ylabel('IG_Salinity*100',fontsize=11)
    # Set y-axis to log scale
    #plt.xlabel('Input feature',fontsize=13)
    #plt.ylabel('feature',fontsize=13)
    #plt.xlabel('composite_shap',fontsize=13)
    plt.subplot(3,1,2)
    plt.plot(years,np.mean(A_SHAP2[:,:,1],axis=1)*100,'g')
    #shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
    plt.ylabel('SHAP_Salinity*100',fontsize=11)
    # Set y-axis to log scale
    #plt.xlabel('Time',fontsize=13)
    #plt.ylabel('feature',fontsize=13)
    #plt.xlabel('composite_shap',fontsize=13)
    plt.subplot(3,1,3)
    plt.plot(years,np.mean(A_deeplift2[:,:,1],axis=1)*100,'b')
    #shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
    plt.ylabel('DL_Salinity*100',fontsize=11)
    # Set y-axis to log scale
    plt.xlabel('Time',fontsize=11)
    plt.savefig("ig_composite_8_NADW_time_total_"+str(w)+".png")

for w in range(1):
    w=2
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(years,np.mean(A_IG2[:,:,2],axis=1)*100,'r')
    #shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
    plt.ylabel('IG_EKE*100',fontsize=11)
    # Set y-axis to log scale
    #plt.xlabel('Input feature',fontsize=13)
    #plt.ylabel('feature',fontsize=13)
    #plt.xlabel('composite_shap',fontsize=13)
    plt.subplot(3,1,2)
    plt.plot(years,np.mean(A_SHAP2[:,:,2],axis=1)*100,'g')
    #shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
    plt.ylabel('SHAP_EKE*100',fontsize=11)
    # Set y-axis to log scale
    #plt.xlabel('Time',fontsize=13)
    #plt.ylabel('feature',fontsize=13)
    #plt.xlabel('composite_shap',fontsize=13)
    plt.subplot(3,1,3)
    plt.plot(years,np.mean(A_deeplift2[:,:,2],axis=1)*100,'b')
    #shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
    plt.ylabel('DL_EKE*100',fontsize=11)
    # Set y-axis to log scale
    plt.xlabel('Time',fontsize=11)
    plt.savefig("ig_composite_8_NADW_time_total_"+str(w)+".png")

for w in range(1):
    w=3
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(years,np.mean(A_IG2[:,:,3],axis=1)*100,'r')
    #shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
    plt.ylabel('IG_N*100',fontsize=11)
    # Set y-axis to log scale
    #plt.xlabel('Input feature',fontsize=13)
    #plt.ylabel('feature',fontsize=13)
    #plt.xlabel('composite_shap',fontsize=13)
    plt.subplot(3,1,2)
    plt.plot(years,np.mean(A_SHAP2[:,:,3],axis=1)*100,'g')
    #shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
    plt.ylabel('SHAP_N*100',fontsize=11)
    # Set y-axis to log scale
    #plt.xlabel('Time',fontsize=13)
    #plt.ylabel('feature',fontsize=13)
    #plt.xlabel('composite_shap',fontsize=13)
    plt.subplot(3,1,3)
    plt.plot(years,np.mean(A_deeplift2[:,:,3],axis=1)*100,'b')
    #shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
    plt.ylabel('DL_N*100',fontsize=11)
    # Set y-axis to log scale
    plt.xlabel('Time',fontsize=11)
    plt.savefig("ig_composite_8_NADW_time_total_"+str(w)+".png")

barWidth = 0.25
years=np.arange(1993,2015,1/12)

vector = np.nanmean(np.nanmean(np.abs(A_IG[:, :, :]), axis=0), axis=0).reshape(4,)
IG = vector.tolist()
vector = np.nanmean(np.nanmean(np.abs(A_SHAP[:, :, :]), axis=0), axis=0).reshape(4,)
SHAP = vector.tolist()
vector = np.nanmean(np.nanmean(np.abs(A_deeplift[:, :, :]), axis=0), axis=0).reshape(4,)
DL = vector.tolist()
br1 = np.arange(len(IG)) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2]


plt.figure()
plt.bar(br1,IG, yerr=A_IG_std,color ='r', width = barWidth, 
        edgecolor ='grey', label ='IG')
plt.bar(br2,SHAP, yerr=A_SHAP_std, color ='g', width = barWidth,
        edgecolor ='grey', label ='SHAP')
plt.bar(br3,DL, yerr=A_deeplift_std, color ='b', width = barWidth,
        edgecolor ='grey', label ='deeplift')
#shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
#plt.ylabel('IG',fontsize=13)
# Set y-axis to log scale
plt.xlabel('Input feature',fontsize=13)
#plt.ylabel('feature',fontsize=13)
#plt.xlabel('composite_shap',fontsize=13)
plt.xticks([r + barWidth for r in range(len(IG))],
        ['PT', 'Salinity', 'EKE', 'N'])
plt.legend()
plt.savefig("XAI_composite_2_NADW_0.png")

vector = np.nanmean(np.nanmean(np.abs(A_IG2[:, :, :]), axis=0), axis=0).reshape(4,)
IG = vector.tolist()
vector = np.nanmean(np.nanmean(np.abs(A_SHAP2[:, :, :]), axis=0), axis=0).reshape(4,)
SHAP = vector.tolist()
vector = np.nanmean(np.nanmean(np.abs(A_deeplift2[:, :, :]), axis=0), axis=0).reshape(4,)
DL = vector.tolist()
br1 = np.arange(len(IG))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]


plt.figure()
plt.bar(br1,IG, yerr=A_IG_std2,color ='r', width = barWidth, 
        edgecolor ='grey', label ='IG')
plt.bar(br2,SHAP, yerr=A_SHAP_std2, color ='g', width = barWidth,
        edgecolor ='grey', label ='SHAP')
plt.bar(br3,DL, yerr=A_deeplift_std2, color ='b', width = barWidth,
        edgecolor ='grey', label ='deeplift')
#shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
#plt.ylabel('IG',fontsize=13)
# Set y-axis to log scale
plt.xlabel('Input feature',fontsize=13)
#plt.ylabel('feature',fontsize=13)
#plt.xlabel('composite_shap',fontsize=13)
plt.xticks([r + barWidth for r in range(len(IG))],
        ['PT', 'Salinity', 'EKE', 'N'])
plt.legend()
plt.savefig("XAI_composite_8_NADW_0.png")


