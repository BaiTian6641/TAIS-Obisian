bioRxiv preprint doi: https://doi.org/10.64898/2026.03.27.714843; this version posted March 31, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made available under aCC-BY-NC-ND 4.0 International license. 

**Sharp wave-ripple clusters enhance hippocampal-neocortical engagement for memory consolidation** 

Mihály Vöröslakos<sup>1†</sup> , Christopher Lafferty<sup>1†</sup> , ZheYang Zheng<sup>1†</sup> , Nicholas Paleologos<sup>1</sup> , Elisa Chinigo<sup>2</sup> , Kathryn McClain<sup>1</sup> , Deren Aykan<sup>1</sup> , Euisik Yoon<sup>3</sup> and György Buzsáki<sup>1,4*</sup> 

1Neuroscience Institute, Grossman School of Medicine, New York University, New York, NY, USA 

2 Center for Neural Science, New York University, New York, NY, USA 

3Department of Electrical Engineering and Computer Science, University of Michigan, Ann Arbor, MI, USA 

4Department of Neurology, Grossman School of Medicine, New York University, 435 East 30th Street, New York, NY, 10016, USA 

†These authors contributed equally to this work. 

*Corresponding author. 

# **Abstract** 

Hippocampal sharp wave-ripples (SPW-Rs), neocortical slow oscillations, and thalamocortical sleep spindles are hypothesized to provide a temporal framework for coordinated information transfer during memory consolidation. Hippocampal replay supports this process, yet replayed sequences often unfold across multiple SPW-Rs, suggesting that individual ripples may not constitute the fundamental unit of hippocampal output. Here, using large-scale electrophysiological recordings from the hippocampus and retrosplenial cortex, we show that hippocampal output is organized into clusters of SPW-Rs (cSPW-Rs) during UP states, which are often phase-locked to spindle troughs. Extending this approach with wide-field imaging and unsupervised latent-variable modeling, we found that cSPW-Rs enhanced segregation between the default mode and somatomotor networks and preferentially replayed spatially extended maze trajectories following learning. We propose that SPW-R clusters enable reverberating hippocampal-cortical spike exchange and the concatenation of sequential experiences, establishing ripple clusters as a previously unrecognized syntactic unit of hippocampal-neocortical dialogue. 

bioRxiv preprint doi: https://doi.org/10.64898/2026.03.27.714843; this version posted March 31, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made available under aCC-BY-NC-ND 4.0 International license. 

# **Introduction** 

During sleep, the brain autonomously restructures network connectivity to support the consolidation of new memories. This process is supported by a hippocampal-neocortical dialogue, thought to rely on coordinated interactions among hippocampal sharp wave-ripples (SPW-Rs), neocortical slow oscillations, and thalamocortical sleep spindles<sup>1,2,11,3–10</sup> . These rhythms provide a temporal framework that aligns hippocampal output with cortical depolarization, creating windows for coordinated spike transfer and synaptic plasticity<sup>2,3,17–24,5,9,11–16</sup> . A striking observation, however, is that the duration of hippocampal replay is often longer than that of SPWRs<sup>4,12,32,23,25–31</sup> . Consequently, many replayed neuronal sequences unfold as continuous, uninterrupted events across ripple boundaries, spanning multiple detected SPW-Rs<sup>25,28,30,33–36</sup> . 

This mismatch in timescales suggests that the single SPW-R may not represent the unit of hippocampal output for consolidation. Instead, temporally clustered SPW-Rs may define a fundamental syntax of hippocampal-neocortical dialogue. Since ripples are commonly treated as discrete events, the defining properties and functional significance of such clustered SPW-Rs have never been directly described. Moreover, if ripple clusters constitute an effective unit of information transfer, their influence should be expressed as coordinated changes across distributed cortical systems. Whether these events preferentially reorganize functional cortical networks remains unknown. To address these questions, we combined high-density electrophysiology with widefield calcium imaging and unsupervised latent-variable modeling to show that clustered ripples broadcast replay of novel experiences while reshaping large-scale cortical dynamics for consolidation. 

# **Results** 

# **Clustered SPW-Rs are prominent during NREM sleep** 

To characterize temporally clustered SPW-Rs, we recorded simultaneously from hippocampal CA1-CA3 regions and retrosplenial cortex (RSC), a major hippocampal output region, using fourshank Neuropixels probes<sup>37</sup> and 128-channel silicon probes ( **Figure 1A, B** ). Inter-SPW-R intervals had a non-uniform distribution with a prominent peak at 122.5 ms (full width at half maximum = 78-182 ms) followed by a long tail ( **Figure 1C** , nSPW-Rs =86354 across 25 sessions in 7 mice). Double Gaussian decomposition in logarithmic space revealed two components of the interval distribution: a fast component capturing temporally clustered SPW-Rs, and a slow component representing isolated events ( **Figure 1C** , bottom, **Figure S1A** ). These components intersected at 177 ms, providing a boundary between clusters and isolated ripples. Notably, the short-interval peak of this fast component (126 ms) was preserved across dorsal, intermediate, and ventral recordings despite regional differences in excitability ( **Figure S1B** , ndorsal SPW-Rs= 44439 and nintermediate/ventral SPW-Rs=94343, in 3 rats), indicating that ripple clusters are a conserved physiological feature along the hippocampal long axis. 

bioRxiv preprint doi: https://doi.org/10.64898/2026.03.27.714843; this version posted March 31, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made available under aCC-BY-NC-ND 4.0 International license. 

To reliably identify these two event types, we defined clustered SPW-Rs ( **c** SPW-Rs) as events separated by <180 ms, and isolated or ‘solo’ SPW-Rs ( **s** SPW-Rs) as those ≥500 ms apart since these thresholds minimized cross-contamination between components ( **Figure S1A,** these thresholds were selected to maximize the separation between the fast and slow inter-event interval components while minimizing cross-contamination, see Methods). By these criteria, 44.9% of ripples occurred in **s** SPW-Rs, and 31.2% were **c** SPW-Rs (23.8% doublets, 5.9% triplets, 1.6% four or more, **Figures S1F, 2** ). **c** SPW-R onsets exhibited an asymmetric cross-correlogram across all SPW-Rs ( **Figure 1E** ), indicating a refractory period preceding **c** SPW-R occurrence. Within **c** SPWRs, power increased from R1 to R2 of a cluster ("potentiation"), and all cluster ripples were significantly larger than **s** SPW-Rs ( **Figure S1C-E** ). We also found that clustering was strongly state-dependent since the ratio of **c** SPW-Rs/ **s** SPW-Rs was more than twofold higher during NREM sleep than waking ( **Figure 1C** ; **Figure S1G-I** ). 

SPW-Rs occurred mainly during the RSC UP state ( **s** SPW-R: 85.9±1.4% and **c** SPW-R: 85.0±2.0%, mean±SEM, n=14 sessions in 4 mice<sup>23</sup> ). **s** SPW-Rs concentrated near the end of the RSC UP state, which returned to a DOWN state immediately (<50 ms) after a large amplitude **s** SPW-Rs ( **Figure 1F-H** , **J-L)**<sup>23</sup> . In contrast, the probability of occurrence of the first ripple of **c** SPW-R was lowest before the DOWN state (red line in **Figure 1L** ). Following the DOWN-UP transition, ripples with the highest probability occurred ~120 ms after UP state onset, and the subsequent ripples of **c** SPWRs occurred at similar intervals during the persisting UP state ( **Figure 1L** ), coupled with prolonged spiking activity of CA1 and CA3 pyramidal cells ( **Figure 1F-H** ), interneurons ( **Figure S3** ) and RSC neurons ( **Figure 1F;** n=14 sessions in 4 mice). 

We treated DOWN states shorter than 100 ms separately to avoid contamination from spindleinduced short silences (e.g., **Figure 1B** ; **Figure S4C** ). These short “DOWN” states corresponded to the peaks of the spindle waves ( **Figure 1B** ), and their troughs coincided with increased probability of SPW-Rs ( **Figure 1L** ). Together, these results suggest that **c** SPW-Rs preferentially occur during prolonged periods of cortical excitability and exhibit rhythmic structure, suggesting entrainment. 

Consistent with this idea, **c** SPW-Rs were coupled to RSC spindle activity (8–12 Hz; **Figure 1D, I; Figure S4A-E** ), co-occurring with spindles at significantly higher rates than **s** SPW-Rs (12.4±1.0% vs 5.0±0.3%, 2.5-fold, 1.6-fold increase, p=0.001, Wilcoxon signed-rank test, Cohen's dz=1.92; 14 sessions in 4 mice). Ripple-containing spindles exhibited significantly higher peak power and were of longer duration than SPW-R-free spindles ( **Figure S4F-I** ). However, **c** SPW-Rs also occurred in the absence of spindles, suggesting a role for additional mechanisms, such as intrinsic resonance of hippocampal circuits<sup>38</sup> . To test this hypothesis, we optically stimulated CA1 parvalbumin interneurons from 2 to 40 Hz and revealed a strong resonance in the 5-12 Hz band for both putative pyramidal cells and interneurons ( **Figure S5** ). Thus, inter-ripple intervals may be paced by a combination of circuit resonance and phase-entrainment by cortical spindles. 



<!-- Start of picture text -->
A B ‘UP’ es a Cc 3000 D , Power (normalized) ><br>NP ‘DOWN’ K a -<br>2. 2.0 wna Nitta Via i» w= as<br>LtSX SPW-R, RSC - Wdheer “¥aat™N/M A / = 8 1500 2] ra— |=Peat<br>Pall‘iMAN, Ec* eencaminainematarinne adi cilrsthal edi19 ° 0 0 1 2 3 xgy” , ),<br>} TTS ~ = WAKE = NREM 5 ofa i<br>wi™ Qa” 300 ESKLE SSFes} EF 1+ 2kOES SSeed =2 2 ooo ' o4 F3 10: \Fd me ihe<br>mm =© 150F-Te| ne= at=] 12See ee aeet gaas ” w2 oOa | <iif<br>ee =: Po4+- /T _4 I ©8 001 = | :<br>mRSC mCA1 =CA3 okt= a= ~ $- = oa 0 _—We |<br>3391.4 Time; (s) 3392.6 10?Inter-Event10" 10°Interval10° (s)102 415 1-05Time0 (cycles)05 1 15<br>E<br>2pre 3 sSPW-R 6 cSPW-R J< U-D D-U<br>‘s 22 | 4 | =925S 0 | 2 —K<br>os 1 | 2 I O Bo 0<br>& QN<br>0 0 “~2 2<br>05 0 05 -05 0 05 -0.5 0 05 05 0 05<br>Time from Ripple Onset (s) Time from Cluster onset (s) Time from U-to-D transition (s) | Time from D-to-U transition (s)<br>F RSC PYR - Solo RSC PYR - Cluster K<br>Firing rate (z-score) Firing rate (z-score) sSPW-R<br>-2 0 2 4 2 0 2 4 x10*<br>Ll —— EE ~<br>=5g 4 elu 1 7 =& PersPanta enMla ines ReeMRMe esc2 yA a aay<br>8 : oP you § baa ee ie eae FSM 8 feeder ae<br>12 7 12 As Oo 5 fPAS crnSe a Ree Dietlie tte<br>G 05 CA1 PYR0  - Solo 05 05 CA1 PYR0  - Cluster as 2Ss RatherpA a, ceend Ke Peaeek,ae<br>1> Firing rate(z-score) 4 2 Firingo  rate (z-score)2 4 oa aiteeSAPS MeiBeals eee,Cs 2 ee:Ls  wheRL eh iaN ingRASPade a<br>6 4 * 4 = 5 Synch tg He eects A eae: ee Uses e ines<br>g 4 4c et. ace een<br>o$ 128 32 8 =G = S 1 BeyeGCe iaV ee PT A ealteaH RyeRAGAEieetcpieuiepy A g tedCaP SeaedSpeedIRA Ree<br>0.5 —_ 12 — 2 2 |& peeSER Rane weBtiie ot§ eaeape ee 9 | poiseASO APR AeAiaAgeaewensCee a |<br>0 05 0.5 0 05 FE eee it Ree A<br>H CA3 PYR - Solo CA3 PYR - Cluster oe eesaa caeeh ooOM = gheeoy<br>49 firing rate (z-score) 4 2 Firing rate0  (z-score)2 4 -0.5 0 05 -0.5 ie} 05s<br>——L— SEE —— Time;  from U-to-D transition“a: (s) Time. from D-to-U transitionPr (s)<br>a VaR) Thaw! ih eet Te "i =<br><c 4 i i r 4 1 4, =<br>§ if B ; 25¢6€«L cSPW-R<br>8 8 ii 8 I re<br>7) ” sm a k anna ahi re : | § x17 @R1 @R2 @R3 @R4+ x10¢<br>0.5Time 0 0.5 0.5 0 05 PoeBieee See ME, | ooen ee aeeet CaNpie ae Seo BRRy arieseeraceoesat aN<br>| from ripple onset (s) Time from cluster onset (s) «= ricerr Ob SenaAae PRAeh fig cio GS|Lh<br>z“oO | | ACS Sajee CRESTerg Dame5. ceeae erestte BreanneSOME Mm eeeearSAS sareecee ger]ea<br>2 05 0 05 05 0 05 {05 0 05 -05 0 0.5<br>Time from ripple onset (s) Time from cluster onset (s) Time from U-to-D transition (s) Time from D-to-U transition (s)<br><!-- End of picture text -->

bioRxiv preprint doi: https://doi.org/10.64898/2026.03.27.714843; this version posted March 31, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made available under aCC-BY-NC-ND 4.0 International license. 

intervals. Bottom: Logarithmic time scale reveals bimodal structure with distinct cluster and solo ripple populations. The red arrow indicates the local peak of the short-latency "cluster" population (intervals < 1 s) at 0.137 s. The black arrow indicates the global peak representing typical solo ripple intervals at 1.387 s. Blue and red curves show kernel density estimates for wake and NREM sleep states, demonstrating statedependent differences in ripple clustering dynamics. **(D)** Cortical LFP (RSC) centered on detected peaks within the bandpass-filtered spindle oscillation (5-15 Hz). Time 0 represents the peak of each individual oscillatory cycle (n=1000 peaks from one session). Mean LFP waveform is overlaid in white (scale bar: 50 µV). Instantaneous frequency: 9.38 ± 2.28 Hz. Time axis normalized to ±2 cycles relative to each peak to account for frequency variability across individual oscillations. **(E)** Probability density of inter-event intervals showing the likelihood of finding neighboring ripples at various time lags. Left: solo ripples are isolated by >500 ms from any neighboring event (by classification criteria). Right: Cluster ripple onsets show dense temporal clustering, with high probability of additional ripples at short latencies (~100-200 ms), reflecting the typical inter-ripple intervals within cluster sequences. Time 0 represents the reference ripple onset. **(F-H)** Population-average firing rates (z-scored) aligned to solo ripple onset (left) and cluster onset (right) for RSC **(F)** , CA1 **(G)** , and CA3 **(H)** . Heatmaps show session-averaged responses (n=14 sessions), and black traces indicate the mean firing rate across all sessions. Within each region, putative pyramidal cells (PYR) are shown. **(I)** Spindle band power (5-15 Hz) time course for solo (blue) versus cluster (orange) ripples (spindle power was Z-scored relative to a baseline -1.0 to -0.8s relative to ripple onset). Left: solo ripples exhibit a brief, transient increase in spindle power centered at ripple onset (time 0). Right: Cluster ripples occur during sustained spindle oscillations, with elevated power beginning before the first ripple and persisting throughout the cluster sequence. Grey bars indicate time windows of significant differences between cluster and solo conditions (cluster-based permutation testing, n=5000 permutations, n=14 sessions). **(J)** RSC multi-unit activity (MUA) surrounding UP-DOWN (left) and DOWN-UP (right) transitions (n=14 sessions in 4 mice). K refers to transient rebound population synchrony at the D-U transition, K-complex or “K”. **(K-L)** Raster plots show individual ripple events aligned to cortical state transitions (time 0) with DOWN states shaded in pink. Rows represent individual transitions sorted by DOWN state duration (40-500 ms), with horizontal green line marking 100 ms duration split. Marginal density plots (overlaid lines) show temporal distribution of ripple occurrence for short (<100 ms, below green line) versus long (>100 ms, above green line) DOWN states. **(K)** Solo ripples aligned to UPto-DOWN (n=52701 transitions, left) and DOWN-to-UP transitions (n=52865 transitions, right). **(L)** Cluster ripples aligned to UP-to-DOWN (left), and DOWN-to-UP transitions (right). Ripples are colorcoded by within-cluster position: R1 (first ripple, red), R2 (gold), R3 (green), R4+ (blue). Marginal density plots (overlaid lines) are shown for R1, R2, and R3. Data from 14 NREM sessions in 4 mice. 

# **Clustered SPW-Rs enhance segregation of neocortical domains** 

Because **c** SPW-Rs were coupled to sustained RSC UP states and entrained by RSC spindles, we hypothesized that these clustered events represent a discrete unit of hippocampal output and therefore should exert coordinated, systems-level effects on additional downstream neocortical regions. To test this directly, we combined widefield imaging of the dorsal neocortex with silicon probe recordings in ipsilateral CA1, to examine the association of **c** SPW-Rs on global cortical activity and functional network structure. In Thy1-GCaMP6f mice (n=5), a 64- or 128-channel silicon probe was lowered through the left hemisphere into right hippocampal CA1, ipsilateral to a thinned-skull cranial window preparation ( **Figure 2A)**<sup>23</sup> . Simultaneous optical and electrophysiological recordings were obtained during 1-2 hours of head-fixed sleep, and sleep quality in this condition was comparable to home-cage sleep<sup>23</sup> . Widefield recordings were hemodynamically corrected<sup>39</sup> (see Methods) and aligned to the Allen Institute’s Common Coordinates Framework before parcellation into 27 regions of interest ( **Figure S6)**<sup>40</sup> . 



<!-- Start of picture text -->
mesoscopic widefield imaging 5 eae network 1<br>0000000008 z-scored median regional e « : i<br>a a ‘SSpult<br>470 nm a extracellular ephys fluorescence ® ra tr<br>4 AUDpot r<br>% ‘ to) 3 5 OS<br>f- F a ,) ] Ms r= 0.97"R vontven<br>f SS KH4} < i antvisHt<br>AIP: -2.06 mm ‘$4. ces © 2 (RSC aFiF) > PEELS<br>AS 1 atcorex| 0<br>D E<br>* 15% dF/F 1 50% F G<br>, A " a ms lt 1s S — sSPW-R g ]<br>ca sSPW-R Ss — cSPW-R Bs /<br>1 7 \ a Sa iy<br>0s \ Se 15 VA<br>Ss © © -2 0 2 4 2 0 2 4 Se<br>eS Ss » Time to ripple peak (s) Time to ripple peak (s) gs gs<br>« SS<br>»<br>H I we J K<br>* it} -0.2 ms “mm 0.2 03 0.12<br>sSPW-R . faDre| ie) oe!ON k/<br>“if eel | | ae | aS 3 Ss fy<br>aJ hghod cSPW-R mali=|| Ea | FatZe 3 — sSPW-R ez= a |yLZ<br>: Se | Ca =a<br>baseline A i fos | bebe | Ps — cSPW-R ) a a oj 7<br>r(2) rz) .> 4- Oo 4 2 3 ee-2 0 2 4<br>Time to ripple peak (s) Time to ripple peak (s) os*<br>g2 FP<br><!-- End of picture text -->

bioRxiv preprint doi: https://doi.org/10.64898/2026.03.27.714843; this version posted March 31, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made available under aCC-BY-NC-ND 4.0 International license. 

of NREM regional pairwise correlation for a single mouse. Regions from **B** are highlighted and canonical networks are labeled. **(D)** Summary of aggregate within- and between-network correlations (Z-transformed) across animals, showing that cortical subregions in different networks are less correlated than subregions in the same network (F(2,10) = 47.27, p < 0.05; post-hoc comparisons of SMN and DMN vs SMN-DMN, t(5) > 9.43*). Network boundaries are highlighted (right). **(E)** Sequence of frames depicting WF activity relative to peak power of **s** SPW-R or first **c** SPW-R in a cluster at t = 0, averaged across ripples (nsSPW-R = 340 ncSPWR = 406) for a single animal. **(F)** SPW-R-triggered median fluorescence of a DMN representative subregion (VISp1) for the same animal. Inset depicts subregion outline. **(G)** Summary of the SPW-R-triggered change in fluorescence for VISp1 across animals (t(4) = 3.70*). ΔdF/F was computed as the difference in mean fluorescence between the green and gray bars in **F** , corresponding to pre- and post-SPW-R periods across ripples for each mouse. **(H)** SPW-R-triggered changes in pairwise regional correlations (Z) were computed using a 1 second sliding window, followed by a baseline subtraction using the -5 to -2 second range prior to the ripple peak. **(I)** Sequence of matrices depicting SPW-R-triggered change in pairwise regional correlations, averaged across ripples for a single animal (same as in **E** ). **(J-K)** SPW-R-triggered change in the balance between within-network DMN and SMN correlations (Z) for the same animal and **(K)** across animals (t(4) = 4.57*). ΔDMN-SMN balance was computed as the difference in the mean traces between the green and gray bars in **J** , corresponding to pre- and post-SPW-R periods across ripples for each mouse. Shaded areas ( **F,J** ) and error bars represent SEM. *p < 0.05. 

# **Learning enhances cSPW-R-centered segregation of the default mode and somatomotor networks** 

Because SPW-Rs contribute to memory consolidation<sup>4,18,27,42</sup> , and **c** SPW-Rs are associated with transient segregation of cortical networks, we asked whether **c** SPW-Rs lead to stronger functional reorganization of DM-SM networks after learning. Mice (n = 3) were first trained to obtain water by running in a T-maze in which only one of two arms was accessible. After 14 days of training (‘ _familiar_ ’), mice were tested in the same maze configuration for 50 trials before the second arm was made accessible (‘ _novel_ ’) and the animal was rewarded for alternating between the left and right arms ( **Figure 3A** ). The mice learned the novel alternation task within the same session ( **Figure 3B** ) and continued to alternate with high proficiency the following day ( **Figure 3C** ) suggesting that an enduring memory of the task structure had formed after introducing the novel arm. To capture the effects of this learning on hippocampal-neocortical communication, we imaged the dorsal cortical surface and recorded from hippocampal CA1 during sleep before and after each behavioral session throughout training, as previously described ( **Figure 2A** ). 

While **c** SPW-Rs remained associated with diverging within-network correlations ( **Figure 2I-K; Figure 3D, E** ), only post-learning **c** SPW-Rs were followed by a significant reduction in betweennetwork DMN-SMN correlations ( **Figure 3E-G; Figure S9** ). This effect persisted even after controlling for changes in sleep depth before and after learning ( **Figure S10** ) and across alternative **c** SPW-R definitions ( **Figure S11)** . In summary, **c** SPW-Rs are associated with learning-dependent functional isolation of key cortical networks. These data suggest that **c** SPW-Rs define transient windows during which hippocampal output may be broadcast with reduced cortical somatomotor interference, thereby creating conditions favorable for consolidation. Under this framework, 



<!-- Start of picture text -->
A B familiar Cc<br>Training. 60+ —— familiarnovel armarm 60) , 10) ,<br>PRE-task JBGMCC7eeem POST-task Days of =~ | a<br>Sleep ne arm : = — no<br>O: Sleep = vw novel day E = } 2<br>m2 , 6 60 ge é i<br>PRE-taskSleep J-Maze POST-task familiarweeks =fs fi K —g &5 30 5 0.5<br>Sleep = 6 0 ; ga } 5 \<br>bdd0008» OBASBBS = ‘a rf= anpost- novel day Xo}= ¢<br>med onl fre mmo pn o velst-novelda 0 0<br>0 5 10 15 20 SoPo Noe<br>Trial rd Qe<br>| ! 0.3 0.2<br>familiar novel day familiar novel day — PRE-task Sleep<br>2 | mt 7 Ar (Z) _ |r POST-task Sleep £N<br>ax J s 0.4 N Oc *<br>zg& i ial a 3&c ssoc o \<br>|<br>a Li . NS2g LoDv \<br>[on Z5 Sz7 \ \<br>o a ag ae of \\<br>ep)x is 5 = O°2 \ \<br>ren °<br>4 .a VU0 . 4 ww.0.3<br>-1 0 1 2 SoS<br>Time Se<br>to cSPW-R peak (s) & &<br>g a)<br><!-- End of picture text -->

bioRxiv preprint doi: https://doi.org/10.64898/2026.03.27.714843; this version posted March 31, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made available under aCC-BY-NC-ND 4.0 International license. 

types on high-density CA1 recordings from a third cohort of mice (18 sessions from 6 animals) performing a T-maze alternation task<sup>43</sup> . First, we  used a hierarchical state space decoder<sup>44</sup> to decode the 2D position of the animal ( **Figure 4A** ) during population synchrony events (i.e. a union of population burst and clustered ripple events) during NREM sleep. A shuffle test determined whether the co-activities of the neurons were well-described by the generative model using a spatial template (“on-manifold” events)<sup>45</sup> . We found that 14% of the SPW-Rs (n = 47877) were significantly better explained than by shuffled chance control. In PRE-task sleep and POST-task sleep, **c** SPW-Rs had a significantly higher fraction of on-manifold events compared to **s** SPW-Rs ( **Figure 4B** ; PRE: **s** SPW-Rs: 8%, **c** SPW-Rs: 9%; POST: **s** SPW-Rs: 12%, **c** SPW-Rs: 30%). The lengths of replayed trajectories (as measured by maximum displacement among continuous segments within a population synchrony window; **Figure 4C** ) and the speed of replay correlated with the number of SPW-Rs contained in the synchronous events ( **Figure S12** ). This effect persisted even after controlling for duration of events, accounting for the large discrepancy between solo and cluster ripples ( **Figure 4C** ). The replayed content also differed depending on the maze zone. We distinguished replay of stationary locations, the reward areas and the delay area where animals paused without locomotion ('within'; **Figure 4D** ), from replay of the maze corridors traversed during ambulation ('outside'; **Figure 4D** ). Replays of stationary zones were more frequent during **s** SPW-Rs, whereas replays of the corridors were more frequent during **c** SPW-Rs, especially during POST-task sleep ( **Figure 4E** ). 

Even at the same maze position, the hippocampal neural representation depended on the animal’s behavior, suggesting that a purely spatial decoder conflates behaviorally distinct population patterns that happen to share the same location<sup>45</sup> . To complement the trajectory-based replay analysis, we asked whether the reactivated population patterns (independent of their spatial content) also differed between **c** SPW-Rs and **s** SPW-Rs. We thus used an unsupervised method, the Jump Latent Variable Model (JumpLVM<sup>45</sup> ; **Figure S13A-B** ), to extract latent variables (i.e. patterns of population activities) during the task-epoch. We then characterized the latent by behavior, decoded the latent during synchronous population events, and performed a similar shuffle test as in the supervised case to select “on-manifold” events for further analysis. The JumpLVM reduces population activities to a 1-D nonlinear latent manifold (discretized into bins) without requiring the latent to correspond to external labels, such as position. Behavioral correlates (templates) are learned from the correlational structure of population activities and the smoothness condition, quantified by the similarity of population firing between adjacent latent bins. Based on how frequently each latent bin  activates during each behavior type (during the task epoch), we scored and classified each latent bin into Immobility, Headscan, and Locomotion ( **Figure 4F** ; **Figures S3, 13C** ). Headscan latents preferred **s** SPW-Rs during both PRE- and POST-task sleep, while Locomotion latents preferred **c** SPW-Rs, but only during POST-task sleep ( **Figure 4G, H** ; **Figures S13D, 14** ). On average, Immobility latents preferred **c** SPW-Rs ( **Figure 4G** ), but there was a large spread ( **Figure 4H** ). Together, our results highlight differences in replay content between **c** SPW-Rs and **s** SPW-Rs: **c** SPW-Rs are associated with replaying long trajectories during locomotion, whereas **s** SPW-Rs are more strongly associated with activity patterns during non- 



<!-- Start of picture text -->
A . : . F ou? 1<br>No ripple Off-manifold Short replay Long in PRE Long in POST ae<br>Displacement=0cm Displacement=0cm Dsplacement=3cm Displacement=24cm Displacement=41cm i = s<br>$45 WN nant hos Nutto “ih,aid: pertot aH : :AN : mings: : EloOo sf geome 0<br>BIA 30 ~ 404 A. 55 , TA 484 A nage Spikes” Gates 91<br>20 AN ARO /\ INS NAN™ « is<br>0— 0 ——— 0 —_ 0 << 0 —S gopi iby 9<br>Time (s) 0.26 0 0.28 0 0.08 0 0.25 0 0.21 3 & = Sos<br>me - a + : a5 . a 4 3 0<br>_ P Fr ; a1<br>8 Soy oo.. ¥ ' | ) |—— elt fe)cl<br>ahs a F + 3El 35 oo<br>B On manifold Cc . |Q § a 0 o So<br>0.4 Clustered:Solo: POST POST >> PRE, PRE, ****** Displacementi = perDisplacement sqrt(time;  bin). G PQSOP<br>= PRE: Clustered > Solo, ** A re rat re Clustered - Solo<br>3 POST:A(Clustered) Clustered > Solo, *** , =20 Pl ~o5 Pi 0.15 he ek oe ee ons, ee<br>© > A(Solo), *** , —E ——= post 2a5 ——= post iyx —s* ee Pa)<br>£02 _= _Z=— = Zao i = hp<br>_— Uf A 0) : 0 ; 8 § 0.00 wad APPS |<br>———-——S—S 0 1 2 3 4 8 0 141 2 3 4 5% \\ 4<br>PRE . . a= AA,<br>Solo POST PREClusteredPOST N ripples N ripples a@ -0.15 =\ 4<br>D Clustered- Solo Preference Index E “a F coe"e727& & &28<br>PRE (Position) Within Cuiside Ss g§<br>015<br>az OE POST => 0.009 4 ns. \ * 0.0018 4 ns. ** SM& s SF&<br>|4i A a38 0.008 Sx= Sey 0.0016 j; H<br>£8] {b 0.00 ee a 0.0014 +. ve f ‘0<br>“Sis rr Tt D> X ] g z —— Immobility<br>t ' waiting 5 Z 0.007 > 295 —— Headscan<br>. :<br>re 15 ; 2pre 2post °°! 2~u{pre Qupost 85F009 — Locomotion<br>te o 3B 8 B S 2 8 oe =<br>sta Within. 9 s5 OD a5 oo"a ga 1 ol jlustered0 sol 1<br>Outside z a 5 5 Preference-IndexSolo<br><!-- End of picture text -->

bioRxiv preprint doi: https://doi.org/10.64898/2026.03.27.714843; this version posted March 31, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made available under aCC-BY-NC-ND 4.0 International license. 

probability within (left) or outside (right) of the circled low-locomotion regions in D, separated into event epochs (PRE or POST) and types (c or s). Each line is one session. Black bars, medians across sessions. **c** SPW-Rs decode events with significantly higher probability during locomotion in corridors (i.e., outside) during POST-task sleep. ( **F-H** ) JumpLVM analysis results. ( **F** ) Left: example activation patterns of different types of latent bins. Each dot marks the 2D position of the animal when that latent bin achieves MAP, colored by the behavior type at that time (blue: Immobility, orange: Headscan, green: Locomotion). Right: behavior score of that latent bin. Even though the headscan latent (middle row) still activates during locomotion occasionally, it occurs much more reliably during headscans (i.e. occuring in most headscan events, but only sparsely during locomotion), which is captured by the behavior score. ( **G** ) Clustered-Solo Preference Index per latent bin, separated into PRE- and POST-task sleeps and latent types. Each colored line is a session; solid black short lines, medians across sessions. Dotted line, no preference for **c** SPW-Rs or **s** SPW-Rs. Headscan latents occur primarily during **s** SPW-Rs, while Locomotion latents gain preference for **c** SPW-Rs. ( **H** ) Behavior scores for each behavior type (color) per latent bin as a function of the c-s Preference Index, aligned and averaged across all sessions. 

# **Discussion** 

Our findings, together with prior work, support a coordinated cascade linking cortical slow oscillations, thalamocortical spindles, and hippocampal SPW-Rs during sleep. A fraction of cortical DOWN-UP transitions occur as highly synchronous K-complexes<sup>46–49</sup> , which can trigger thalamocortical spindles<sup>50–52</sup> . The synchronous spiking associated with these spindles can, in turn, entrain clustered SPW-Rs that are phase-locked to spindle troughs ( **Figure 1D** )<sup>3,5,23,53,54</sup> . During the UP state, sustained neocortical activity and SPW-R clustering enable reverberating hippocampal-cortical spike exchange<sup>3,55,56</sup> ; providing a window for structured information transfer and the sequential ordering of cortical representations<sup>57</sup> . This communication window is terminated when the final ripple of a cluster, or a large-amplitude **s** SPW-R drives the network back into a DOWN state ( **Figure 1K, L** )<sup>23,58</sup> . Following learning, these clustered events are accompanied by enhanced segregation of cortical default mode and somatomotor networks (DMN and SMN) ( **Figure 3)** , and by longer, more continuous replay trajectories ( **Figure 4** )<sup>20,28</sup> , consistent with the concatenation of waking experience into coherent episodic sequences<sup>20,59,60</sup> . 

The ripple cluster hypothesis builds on prior work and unifies several observations. **c** SPW-Rs transiently enhance the functional segregation of SMN and DMN, effectively deepening an offline cortical state in which associative regions such as the hippocampus and DMN are privileged while somatomotor areas are disengaged. This pattern mirrors the task-negative organization of the DMN<sup>61</sup> and suggests that ripple clusters create protected windows for hippocampal-neocortical dialogue, minimizing interference from ongoing sensory processing<sup>62,63</sup> . For example, presenting sounds during sleep biases subsequent hippocampal replay content but simultaneously suppresses overall ripple incidence<sup>56</sup> , while minor local body movements during NREM microarousals can robustly suppress SPW-R rates<sup>64</sup> . These data suggest that both intrinsically and extrinsically driven cortical activity can influence ongoing hippocampal-DMN dynamics, necessitating a mechanism that biases cortical dynamics away from bottom-up interference, particularly during early consolidation following novel experiences. From a complementary learning systems perspective, 

bioRxiv preprint doi: https://doi.org/10.64898/2026.03.27.714843; this version posted March 31, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made available under aCC-BY-NC-ND 4.0 International license. 

transient network segregation may also protect against catastrophic interference by restricting when and where hippocampal information is broadcast to the cortex. By limiting cross-network coupling during early consolidation, **c** SPW-Rs could enable the gradual integration of new traces without disrupting established cortical representations<sup>54,65,66</sup> . SPW-R-coupled DOWN states may provide a protected period for cortical consolidation, preventing interference from subsequent hippocampal inputs while synaptic modifications induced by the preceding cluster are stabilized. Thus, DOWN states may help in segregating distinct memory traces during consolidation. 

Several lines of convergent mechanistic evidence suggest that slow oscillations and thalamocortical spindles actively entrain hippocampal ripples, thereby promoting memory consolidation. In humans and other mammals, K-complexes can be triggered by sensory stimulation during NREM sleep at modality-specific sites<sup>52</sup> , and auditory closed-loop entrainment of slow oscillations enhances memory (“targeted memory reactivation”;<sup>9,12,67</sup> , but see Henin et. al. 2019<sup>68</sup> ). Similarly, artificial coupling of SPW-Rs to slow oscillations strengthens cortical responses to ripples<sup>17</sup> , while optogenetic induction of spindles during the UP state reliably triggers hippocampal SPW-Rs<sup>69</sup> , with both manipulations enhancing hippocampus-dependent memory consolidation. Consistent with this role, spindles structure the temporal organization of SPW-Rs, as ripples are phase-locked to spindle troughs ( **Figure 1D** )<sup>2,3,5,11,70</sup> and contribute to a prominent peak in the ripple-autocorrelogram, superimposed on an otherwise stochastic distribution ( **Figure 1C** )<sup>20</sup> . Intrinsic resonant properties of hippocampal circuits may further amplify this entrainment ( **Figure S5** ), contributing to the increased amplitudes observed within **c** SPW-Rs ( **Figure S1C** ) and to the stronger association between large-amplitude SPW-Rs and successful consolidation<sup>71</sup> . Finally, this spindle-ripple coupling is likely mediated through the entorhinal cortex, since UP states and spindles entrain SPW-Rs<sup>53,72</sup> , and layer 3 inactivation reduces SPW-R incidence<sup>20</sup> . 

**c** SPW-Rs may support the binding of sequential experiences into extended episodes, as the intervals between clustered ripples fall within the time window of NMDA receptor-dependent plasticity<sup>73</sup> . This temporal structure could allow consecutive events to be linked and associations to form across ripples. Beyond simple sequencing, **c** SPW-Rs may also integrate new memories into existing knowledge<sup>60</sup> through hippocampal-neocortical reverberation during extended UP states. Consistent with this integrative role, behavioral replay is differentially organized across ripple types. Locomotor trajectories preferentially occur during **c** SPW-Rs and increase after learning, whereas head-scanning and immobility sequences are primarily expressed during **s** SPWRs, and dominate before learning. While debates about 'preplay' and 'replay' have focused on awake ripples and prospective vs. retrospective coding<sup>30,34,74</sup> , our findings reveal analogous heterogeneity within sleep-associated SPW-Rs. Just as awake ripples show functional diversity linked to behavioral demands, sleep ripples segregate into distinct subtypes ( **c** SPW-Rs and **s** SPWRs) that engage cortical networks differentially and may consolidate different behavioral experiences ( **Figure 4** ). This suggests that state-dependent specialization (wake vs. sleep) and pattern-dependent specialization (clustered vs. solo) operate as complementary organizing 

bioRxiv preprint doi: https://doi.org/10.64898/2026.03.27.714843; this version posted March 31, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made available under aCC-BY-NC-ND 4.0 International license. 

mechanisms for hippocampal-neocortical communication.  Overall, these findings indicate that ripple clustering organizes hippocampal output into temporally extended, sequential structures through coordinated interactions with thalamocortical spindles, providing a substrate for compositional computation and abstraction<sup>57,75</sup> . 

bioRxiv preprint doi: https://doi.org/10.64898/2026.03.27.714843; this version posted March 31, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made available under aCC-BY-NC-ND 4.0 International license. 

# **Data availability** : 

All data are available in the manuscript or the supplementary materials or are publicly available in the Buzsaki Lab Databank, https://buzsakilab.com/wp/database/. 

# **Code availability:** 

All code used for data pre-processing is available at https://github.com/buzsakilab/buzcode. The code for analyzing neural data, visualization, and application of the supervised neuronal classifier will be available at Zenodo after manuscript acceptance. 

# **Acknowledgements** 

We thank Joaquin Gonzalez, Gergely Komlósi, JingJing Liu, Heechul Jun and members of our laboratory for helpful comments on the project. 

# **Author contributions:** 

Conceptualization: M.V., G.B.; Funding acquisition: G.B., E.Y., C.L., N.P.; Investigation: M.V., C.L., Z.Z., N.P., E.C., K.M., D.A. Project administration: M.V., G.B.; Supervision: M.V., C.L., G.B.; Visualization: M.V., C.L., Z.Z. Writing – original draft: G.B., M.V.; Writing – review & editing: G.B., M.V., C.L., Z.Z and all other authors. 

# **Funding** : 

This study was supported by National Institutes of Health grants RO1MH122391, RO1MH139216, and U19NS107616 (G.B.), the Simons Foundation (C.L.), Cure for Epilepsy and Seizures grant (N.P.), and NS133978, NS142069 (E. Y.). 

**Competing interests** : The authors declare no competing interests. 

bioRxiv preprint doi: https://doi.org/10.64898/2026.03.27.714843; this version posted March 31, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made available under aCC-BY-NC-ND 4.0 International license. 

# **References** 

1. Helfrich, R.F., Lendner, J.D., Mander, B.A., Guillen, H., Paff, M., Mnatsakanyan, L., Vadera, S., Walker, M.P., Lin, J.J., and Knight, R.T. (2019). Bidirectional prefrontalhippocampal dynamics organize information transfer during sleep in humans. Nat. Commun., 1–16. 10.1038/s41467-019-11444-x. 

2. Siapas, A.G., and Wilson, M.A. (1998). Coordinated Interactions between Hippocampal Ripples and Cortical Spindles during Slow-Wave Sleep. Neuron _21_ , 1123–1128. 

3. Sirota, A., Csicsvari, J., Buhl, D., and Buzsaki, G. (2003). Communication between neocortex and hippocampus during sleep in rodents. Proc. Natl. Acad. Sci. _100_ , 2065– 2069. 10.1073/pnas.0437938100. 

4. Buzsáki, G. (2015). Hippocampal sharp wave-ripple: A cognitive biomarker for episodic memory and planning. Hippocampus _25_ , 1073–1188. 10.1002/hipo.22488. 

5. Staresina, B.P., Bergmann, T.O., Bonnefond, M., Meij, R. Van Der, Jensen, O., Deuker, L., Elger, C.E., Axmacher, N., and Fell, J. (2015). Hierarchical nesting of slow oscillations , spindles and ripples in the human hippocampus during sleep. Nat. Neurosci. _18_ . 10.1038/nn.4119. 

6. Astori, S., Wimmer, R.D., and Lüthi, A. (2013). Manipulating sleep spindles – expanding views on sleep , memory , and disease. Trends i _36_ . 10.1016/j.tins.2013.10.001. 

7. Niethard, N., Ngo, H. V, Ehrlich, I., and Born, J. (2018). Cortical circuit activity underlying sleep slow oscillations and spindles. PNAS _115_ . 10.1073/pnas.1805517115. 

8. Seibt, J., Richard, C.J., Sigl-glöckner, J., Takahashi, N., Kaplan, D.I., Doron, G., Limoges, D. De, Bocklisch, C., and Larkum, M.E. (2017). Cortical dendritic activity correlates with spindle-rich oscillations during sleep in rodents. Nat. Commun., 1–12. 10.1038/s41467017-00735-w. 

9. Jiang, X.X., Gonzalez-Martinez, J., and Halgren, E. (2019). Coordination of Human Hippocampal Sharpwave Ripples during NREM Sleep with Cortical Theta Bursts, Spindles, Downstates, and Upstates. J. Neurosci. _39_ , 8744–8761. 

10. Klinzing, J.G., Mölle, M., Weber, F., Supp, G., Hipp, J.F., Engel, A.K., and Born, J. (2016). Spindle activity phase-locked to sleep slow oscillations. Neuroimage _134_ , 607– 616. 10.1016/j.neuroimage.2016.04.031. 

11. Klinzing, J.G., Niethard, N., and Born, J. (2019). Mechanisms of systems memory consolidation during sleep. Nat. Neurosci. _22_ , 1598–1610. 10.1038/s41593-019-0467-3. 

12. Ngo, H., Fell, J., and Staresina, B. (2020). Sleep spindles mediate hippocampalneocortical coupling during long-duration ripples. Elife, 1–18. 

13. Buzsáki, G. (1998). Memory consolidation during sleep : a neurophysiological perspective. J. Sleep Res. _7_ , 17–23. 

14. Andrade, C., Spoormaker, V.I., Dresler, M., Wehrle, R., Holsboer, F., Sa, P.G., and Czisch, M. (2011). Sleep Spindles and Hippocampal Functional Connectivity in. J. 

bioRxiv preprint doi: https://doi.org/10.64898/2026.03.27.714843; this version posted March 31, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made available under aCC-BY-NC-ND 4.0 International license. 

# Neurosci. _31_ , 10331–10339. 10.1523/JNEUROSCI.5660-10.2011. 

15. Battaglia, F.P., Sutherland, G.R., and Mcnaughton, B.L. (2004). Hippocampal sharp wave bursts coincide with neocortical “ up-state ” transitions. Learn. Mem. _11_ , 697–704. 10.1101/lm.73504.anesthetized. 

16. Mölle, M., Yeshenko, O., Marshall, L., Sara, S.J., and Born, J. (2006). Hippocampal Sharp Wave-Ripples Linked to Slow Oscillations in Rat Slow-Wave Sleep. J. Neurophysiol. _96_ , 62–70. 10.1152/jn.00014.2006.Slow. 

17. Maingret, N., Girardeau, G., Todorova, R., Goutierre, M., and Zugaro, M. (2016). Hippocampo-cortical coupling mediates memory consolidation during sleep. Nat. Neurosci. _19_ , 959–964. 10.1038/nn.4304. 

18. Wilson, M.A., and Mcnaughton, B. (1994). Reactivation of Hippocampal Ensemble Memories During Sleep. Science (80-. ). _265_ , 676–679. 

19. Cowan, E., Liu, A., Henin, S., Kothare, S., Devinsky, O., and Davachi, L. (2020). Sleep Spindles Promote the Restructuring of Memory Representations in Ventromedial Prefrontal Cortex through Enhanced Hippocampal – Cortical Functional Connectivity. J. Neurosci. _40_ , 1909–1919. 

20. Yamamoto, J., and Tonegawa, S. (2017). Direct Medial Entorhinal Cortex Input to Hippocampal CA1 Is Crucial for Extended Quiet Awake Replay. Neuron _96_ , 217–227. 10.1016/j.neuron.2017.09.017. 

21. Fechner, J., Contreras, M.P., Zorzo, C., Shan, X., Born, J., and Inostroza, M. (2024). Sleep-slow oscillation-spindle coupling precedes spindle-ripple coupling during development. Sleep _47_ , 1–13. 

22. Peyrache, A., Khamassi, M., Benchenane, K., Wiener, S.I., and Battaglia, F.P. (2009). Replay of rule-learning related neural patterns in the prefrontal cortex during sleep. Nat. Neurosci. _12_ . 10.1038/nn.2337. 

23. Swanson, R.A., Chinigo, E., Levenstein, D., Voroslakos, M., Mousavi, N., Wang, X., Basu, J., and Buzsaki, G. (2025). Topography of putative bi-directional interaction between hippocampal sharp-wave ripples and neocortical slow oscillations. Neuron _113_ , 754–768. 10.1016/j.neuron.2024.12.019. 

24. Oyanedel, C.N., and Born, J. (2020). Temporal associations between sleep slow oscillations, spindles and ripples. Eur. J. Neurosci., 4762–4778. 10.1111/ejn.14906. 

25. Fernández‐Ruiz, A., Oliva, A., Oliveira, E.F. De, Rocha-Almeida, F., Tingley, D., and Buzsaki, G. (2019). Long-duration hippocampal sharp wave ripples improve memory. Science (80-. ). _364_ , 1082–1086. 

26. Nádasdy, Z., Hirase, H., Czurkó, A., Csicsvari, J., and Buzsáki, G. (1999). Replay and time compression of recurring spike sequences in the hippocampus. J. Neurosci. _19_ , 9497– 9507. 10.1523/jneurosci.19-21-09497.1999. 

27. Lee, A.K., and Wilson, M.A. (2002). Memory of sequential experience in the hippocampus during slow wave sleep. Neuron _36_ , 1183–1194. 10.1016/S0896- 

bioRxiv preprint doi: https://doi.org/10.64898/2026.03.27.714843; this version posted March 31, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made available under aCC-BY-NC-ND 4.0 International license. 

6273(02)01096-6. 

28. Davidson, T.J., Kloosterman, F., and Wilson, M.A. (2009). Hippocampal Replay of Extended Experience. Neuron _63_ , 497–507. 10.1016/j.neuron.2009.07.027. 

29. Foster, D.J., and Wilson, M.A. (2006). Reverse replay of behavioural sequences in hippocampal place cells during the awake state. Nature _440_ , 680–683. 10.1038/nature04587. 

30. Silva, D., Feng, T., and Foster, D.J. (2015). Trajectory events across hippocampal place cells require previous experience. Nat. Neurosci. _18_ , 1772–1779. 10.1038/nn.4151. 

31. Stella, F., Baracskay, P., Neill, J.O., and Csicsvari, J. (2019). Hippocampal Reactivation of Random Trajectories Article Hippocampal Reactivation of Random Trajectories Resembling Brownian Diffusion. Neuron _102_ , 450–461. 10.1016/j.neuron.2019.01.052. 

32. Mallory, C.S., Widloski, J., and Foster, D.J. (2025). The time course and organization of hippocampal replay. Science (80-. ). _548_ , 541–548. 10.1126/science.ads4760. 

33. Pfeiffer, B.E., and Foster, D.J. (2013). Hippocampal place-cell sequences depict future paths to remembered goals. Nature _497_ , 74–79. 10.1038/nature12112. 

34. Grosmark, A.D., and Buzsáki, G. (2016). Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences. Science (80-. ). _351_ , 1440–1443. 10.1126/science.aad1935. 

35. Pfeiffer, B.E. (2020). The content of hippocampal “ replay .” Hippocampus _30_ , 6–18. 10.1002/hipo.22824. 

36. Chen, Z.S., and Wilson, M.A. (2023). Now and Then How our understanding of memory replay evolves. J. Neurophysiol. _129_ , 552–580. 10.1152/jn.00454.2022. 

37. Steinmetz, N.A., Aydin, C., Lebedeva, A., Okun, M., Pachitariu, M., Bauza, M., Beau, M., Bhagat, J., Böhm, C., Broux, M., et al. (2021). Neuropixels 2.0: A miniaturized highdensity probe for stable, long-term brain recordings. Science (80-. ). _372_ , eabf4588. 10.1126/science.abf4588. 

38. Stark, E., Eichler, R., Roux, L., Fujisawa, S., Rotstein, H.G., and Buzsáki, G. (2013). Inhibition-Induced theta resonance in cortical circuits. Neuron _80_ , 1263–1276. 10.1016/j.neuron.2013.09.033. 

39. Ma, Y., Shaik, M.A., Kim, S.H., Kozberg, M.G., Thibodeaux, D.N., Zhao, H.T., Yu, H., and Hillman, E.M.C. (2016). Wide-field optical mapping of neural activity and brain haemodynamics : considerations and novel approaches. Philos. Trans. R. Soc. London. 

40. Wang, Q., Ding, S., Li, Y., Zeng, H., Harris, J.A., Ng, L., Wang, Q., Ding, S., Li, Y., Royall, J., et al. (2020). The Allen Mouse Brain Common Coordinate Framework : A 3D Reference Atlas. Cell _181_ , 936–953. 10.1016/j.cell.2020.04.007. 

41. Zingg, B., Hintiryan, H., Gou, L., Song, M.Y., Bay, M., Bienkowski, M.S., Foster, N.N., Yamashita, S., Bowman, I., Toga, A.W., et al. (2014). Resource Neural Networks of the Mouse Neocortex. Cell _156_ , 1096–1111. 10.1016/j.cell.2014.02.023. 

bioRxiv preprint doi: https://doi.org/10.64898/2026.03.27.714843; this version posted March 31, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made available under aCC-BY-NC-ND 4.0 International license. 

42. Jadhav, S.P., Kemere, C., German, P.W., and Frank, L.M. (2012). Awake Hippocampal Sharp-Wave Ripples Support Spatial Memory. Science (80-. ). _336_ , 1454–1458. 10.1126/science.1217230. 

43. Huszár, R., Zhang, Y., Blockus, H., and Buzsáki, G. (2022). Preconfigured dynamics in the hippocampus are guided by embryonic birthdate and rate of neurogenesis. Nat. Neurosci. _25_ , 1201–1212. 10.1038/s41593-022-01138-x. 

44. Denovellis, E.L., Gillespie, A.K., Coulter, M.E., Sosa, M., Chung, J.E., Eden, U.T., and Frank, L.M. (2021). Hippocampal replay of experience at real- world speeds. Elife, 1–33. 

45. Zheng, Z.S., Zutshi, I., Huszár, R., Zhang, Y., Karadas, M., and Buzsáki, G. (2025). From labels to latents : revealing state-dependent hippocampal computations with Jump Latent Variable Model. bioRxiv. 

46. Davis, H., Davis, P.A., Loomis, A.L., Harvey, E.N., and Hobart, G. (1939). Electrical reactions of the human brain to auditory stimulation during sleep. J. Neurophysiol. _2_ . 

47. Loomis, A.L., Harvey, E.N., and Hobart, G.A. (1938). Distribution of disturbance-patterns in the human electroencephalogram with special reference to sleep. J. Neurophysiol. _1_ , 413–430. 

48. Cash, S.S., Halgren, E., Dehghani, N., Rossetti, A.O., Thesen, T., Wang, C., Devinsky, O., Kuzniecky, R., Doyle, W., Madsen, J.R., et al. (2009). The Human K-Complex Represents an Isolated Cortical Down-State. Science (80-. ). _324_ , 1084–1088. 

49. Amzica, F., and Steriade, M. (1997). The K-complex : Its slow (<1-Hz) rhythmicity and relation to delta waves. Neurology _49_ , 952–959. 

50. Steriade, M., Mccormick, D.A., and Sejnowski, T.J. (1993). Thalamocortical Oscillations in the Sleeping and Aroused Brain. Science (80-. ). _262_ , 679–685. 

51. Fernandez, L.M.J., and Lüthi, A. (2020). Sleep spindles: mechanisms and functions. Physiol. Rev. _100_ , 805–868. 10.1152/physrev.00042.2018. 

52. Gennaro, L. De, and Ferrara, M. (2003). Sleep spindles : an overview. Sleep Med. Rev. _7_ , 423–440. 10.1016/S1087-0792(02)00116-8. 

53. Isomura, Y., Sirota, A., Özen, S., Montgomery, S., Mizuseki, K., Henze, D.A., and Buzsáki, G. (2006). Integration and segregation of activity in entorhinal-hippocampal subregions by neocortical slow oscillations. Neuron _52_ , 871–882. 10.1016/j.neuron.2006.10.023. 

54. Diekelmann, S., and Born, J. (2010). The memory function of sleep. Nat. Rev. Neurosci. _11_ , 114–126. 10.1038/nrn2762. 

55. Buzsaki, G. (1996). The Hippocampo-Neocortical Dialogue. Cereb. cortex _6_ , 81–92. 

56. Rothschild, G., Eban, E., and Frank, L.M. (2017). A cortical – hippocampal – cortical loop of information processing during memory consolidation. Nat. Neurosci. _20_ . 10.1038/nn.4457. 

57. Kurth-nelson, Z., Behrens, T., Wayne, G., Miller, K., Luettgau, L., Dolan, R., Liu, Y., and 

bioRxiv preprint doi: https://doi.org/10.64898/2026.03.27.714843; this version posted March 31, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made available under aCC-BY-NC-ND 4.0 International license. 

Schwartenbeck, P. (2023). Perspective Replay and compositional computation. Neuron _111_ , 454–469. 10.1016/j.neuron.2022.12.028. 

58. Peyrache, A., Battaglia, F.P., and Destexhe, A. (2011). Inhibition recruitment in prefrontal cortex during sleep spindles and gating of hippocampal inputs. PNAS _108_ . 10.1073/pnas.1103612108. 

59. Wu, X., and Foster, D.J. (2014). Hippocampal Replay Captures the Unique Topological Structure of a Novel Environment. J. Neurosci. _34_ , 6459–6469. 10.1523/JNEUROSCI.3414-13.2014. 

60. Liu, Y., Dolan, R.J., Kurth-Nelson, Z., and Behrens, T.E.J. (2019). Human Replay Spontaneously Reorganizes Experience. Cell _178_ , 640–652. 10.1016/j.cell.2019.06.012. 

61. Shulman, G.L., Fiez, J.A., Corbetta, M., Buckner, R.L., Miezin, F.M., Raichle, M.E., and Petersen, S.E. (1997). Common Blood Flow Changes across Visual Tasks: II. Decreases in Cerebral Cortex. J. Cogn. Neurosci. _9_ , 648–663. 10.1162/jocn.1997.9.5.648. 

62. Khodagholy, D., Gelinas, J.N., and Buzsaki, G. (2017). Learning-enhanced coupling between ripple oscillations in association cortices and hippocampus. Science (80-. ). _358_ , 369–372. 

63. Huijbers, W., Pennartz, C.M.A., Cabeza, R., and Daselaar, S.M. (2011). The Hippocampus Is Coupled with the Default Network during Memory Retrieval but Not during Memory Encoding. PLoS One _6_ . 10.1371/journal.pone.0017463. 

64. Rios, A., Usui, M., and Isomura, Y. (2025). Modulation of Hippocampal Sharp-Wave Ripples by Behavioral States and Body Movements in Head-Fixed Rodents. eNeuro _12_ , 1–15. 

65. McClelland, J.L., Mcnaughton, B.L., and Reilly, R.C.O. (1995). Why There Are Complementary Learning Systems in the Hippocampus and Neocortex : Insights From the Successes and Failures of Connectionist Models of Learning and Memory. Psychol. Rev. _102_ , 419–457. 

66. Schapiro, A.C., Mcdevitt, E.A., Chen, L., Norman, K.A., Mednick, S.C., and Rogers, T.T. (2017). Sleep Benefits Memory for Semantic Category Structure While Preserving Exemplar-Specific Information. Sci. Rep., 1–13. 10.1038/s41598-017-12884-5. 

67. Lewis, P.A., and Bendor, D. (2019). Minireview How Targeted Memory Reactivation Promotes the Selective Strengthening of Memories in Sleep. Curr. Biol. _29_ , 906–912. 10.1016/j.cub.2019.08.019. 

68. Henin, S., Borges, H., Shankar, A., Sarac, C., Melloni, L., Flinker, A., Parra, L.C., Buzsaki, G., Devinsky, O., and Liu, A. (2019). Closed-Loop Acoustic Stimulation Enhances Sleep Oscillations But Not Memory Performance. eNeuro _6_ . 

69. Latchoumane, C.-F. V., Ngo, H.-V. V., Born, J., and Shin, H.-S. (2017). Thalamic Spindles Promote Memory Formation during Sleep through Triple Phase-Locking of. Neuron _95_ , 424–435. 10.1016/j.neuron.2017.06.025. 

70. Clemens, Z., Molle, M., Eross, L., Barsi, P., Halasz, P., and Born, J. (2007). Temporal 

bioRxiv preprint doi: https://doi.org/10.64898/2026.03.27.714843; this version posted March 31, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made available under aCC-BY-NC-ND 4.0 International license. 

coupling of parahippocampal ripples, sleep spindles and slow oscillations in humans. Brain _130_ , 2868–2878. 10.1093/brain/awm146. 

71. Robinson, H.L., Todorova, R., Nagy, G.A., Gruzdeva, A., Paudel, P., Oliva, A., and Fernandez-ruiz, A. (2026). Large sharp-wave ripples promote hippocampo- cortical memory reactivation and consolidation during sleep. Neuron _114_ , 226-236.e6. 10.1016/j.neuron.2025.10.003. 

72. Sullivan, D., Csicsvari, J., Mizuseki, K., Montgomery, S., Diba, K., and Buzsaki, G. (2011). Relationships between Hippocampal Sharp Waves, Ripples, and Fast Gamma Oscillation: Influence of Dentate and Entorhinal Cortical Activity. J. Neurosci. _31_ , 8605– 8616. 10.1523/JNEUROSCI.0294-11.2011. 

73. Malenka, R.C., and Nicoll, R.A. (1993). NMDA-receptor-dependent synaptic plasticity : multiple forms andmechanisms. Trends Neurosci. _16_ , 521–527. 

74. Dragoi, G., and Tonegawa, S. (2011). Preplay of future place cell sequences by hippocampal cellular assemblies. Nature _469_ , 397–401. 10.1038/nature09633. 

75. Schwartenbeck, P., Baram, A., Liu, Y., Mark, S., Muller, T., Dolan, R., Botvinick, M., Kurth-Nelson, Z., and Behrens, T. (2023). Generative replay underlies compositional inference in the hippocampal-prefrontal circuit. Cell _186_ , 4885–4897. 10.1016/j.cell.2023.09.004. 

76. Gonzalez, J., Vöröslakos, M., Aykan, D., Soto, N., Nitzan, N., Swanson, R., Chen, Z.S., and Buzsáki, G. (2026). Subspace communication in the hippocampal-retrosplenial axis. bioRxiv. 

77. Vöröslakos, M., Petersen, P.C., Vöröslakos, B., and Buzsáki, G. (2021). Metal microdrive and head cap system for silicon probe recovery in freely moving rodent. Elife _10_ , 1–21. 10.7554/eLife.65859. 

78. Osborne, J.E., and Dudman, J.T. (2014). RIVETS: A mechanical system for in vivo and in vitro electrophysiology and imaging. PLoS One _9_ , 1–10. 10.1371/journal.pone.0089007. 

79. Angotzi, G.N., Vöröslakos, M., Perentos, N., Ribeiro, J.F., Vincenzi, M., Boi, F., Lecomte, A., Orban, G., Genewsky, A., Schwesig, G., et al. (2025). Multi‐Shank 1024 Channels Active SiNAPS Probe for Large Multi‐Regional Topographical Electrophysiological Mapping ofNeural Dynamics. Adv. Sci. _12_ . 

80. Stark, E., Koos, T., and Buzsáki, G. (2012). Diode probes for spatiotemporal optical control of multiple neurons in freely moving animals. J. Neurophysiol. _108_ , 349–363. 10.1152/jn.00153.2012. 

81. Kvitsiani, D., Ranade, S., Hangya, B., Taniguchi, H., Huang, J.Z., and Kepecs, A. (2013). Distinct behavioural and network correlates of two interneuron types in prefrontal cortex. Nature _498_ , 363–366. 10.1038/nature12176. 

82. Peters, A.J., Fabre, J.M.J., Steinmetz, N.A., Harris, K.D., and Carandini, M. (2021). Striatal activity topographically reflects cortical activity. Nature _591_ . 10.1038/s41586020-03166-8. 

bioRxiv preprint doi: https://doi.org/10.64898/2026.03.27.714843; this version posted March 31, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made available under aCC-BY-NC-ND 4.0 International license. 

83. Petersen, P.C., Vöröslakos, M., and Buzsaki, G. (2022). Brain temperature affects quantitative features of hippocampal sharp wave ripples. J. Neurophysiol., 1417–1425. 10.1152/jn.00047.2022. 

84. Watson, B.O., Levenstein, D., Greene, J.P., Gelinas, J.N., and Buzsáki, G. (2016). Network Homeostasis and State Dynamics of Neocortical Sleep. Neuron _90_ , 839–852. 10.1016/j.neuron.2016.03.036. 

85. Johnson, L.A., Blakely, T., Hermes, D., Hakimian, S., Ramsey, N.F., and Ojemann, J.G. (2012). Sleep spindles are locally modulated by training on a brain – computer interface. PNAS _109_ , 18583–18588. 10.1073/pnas.1207532109. 

86. Sullivan, D., Mizuseki, K., Sorgi, A., and Buzsáki, G. (2014). Comparison of Sleep Spindles and Theta Oscillations in the Hippocampus. J. Neurosci. _34_ , 662–674. 10.1523/JNEUROSCI.0552-13.2014. 

87. Bandarabadi, M., Herrera, C.G., Gent, T.C., Bassetti, C., Schindler, K., and Adamantidis, A.R. (202AD). A role for spindles in the onset of rapid eye movement sleep. Nat. Commun. 10.1038/s41467-020-19076-2. 

88. Pachitariu, M., Sridhar, S., and Stringer, C. (2023). Solving the spike sorting problem with Kilosort. bioRxiv. 10.1101/2023.01.07.523036. 

89. Petersen, P.C., Siegle, J.H., Steinmetz, N.A., Mahallati, S., and Buzsáki, G. (2021). CellExplorer: A framework for visualizing and characterizing single neurons. Neuron _109_ , 3594-3608.e2. 10.1016/j.neuron.2021.09.002. 

90. Muller, R.U., and Kubie, J.L. (1987). The effects of changes in the environment on the spatial firing of hippocampal complex-spike cells. J. Neurosci. _7_ , 1951–1968. 10.1523/jneurosci.07-07-01951.1987. 

91. Lever, C., Wills, T., Cacucci, F., Burgess, N., and Keefe, J.O. (2002). Long-term plasticity in hippocampal place-cell representation of environmental geometry. Nature _416_ , 236– 238. 

92. Alvernhe, A., Save, E., and Poucet, B. (2011). Local remapping of place cell firing in the Tolman detour task. Eur. J. Neurosci. _33_ , 1696–1705. 10.1111/j.1460-9568.2011.07653.x. 

93. Monaco, J.D., Rao, G., Roth, E.D., and Knierim, J.J. (2014). Attentive scanning behavior drives one-trial potentiation of hippocampal place fields. Nat. Neurosci. _17_ . 10.1038/nn.3687. 

94. Zheng, Z.S., Huszar, R., Hainmueller, T., Bartos, M., Williams, A.H., and Buzsáki, G. (2024). Perpetual step-like restructuring of hippocampal circuit dynamics. Cell Rep. _43_ . 10.1016/j.celrep.2024.114702. 

95. Lab, L.F. (2024). Track linearization: 2d to 1d position linearization using hmms. Github. https://github.com/LorenFrankLab/track_linearization. 

96. Yang, W., Sun, C., Huszár, R., Hainmueller, T., Kiselev, K., and Buzsáki, G. (2024). Selection of experience for memory by hippocampal sharp wave ripples. Science (80-. ). _383_ , 1478–1483. 10.1126/science.adk8261. 

bioRxiv preprint doi: https://doi.org/10.64898/2026.03.27.714843; this version posted March 31, 2026. The copyright holder for this preprint (which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made available under aCC-BY-NC-ND 4.0 International license. 

97. Gillespie, A.K., Maya, D.A.A., Denovellis, E.L., Liu, D.F., Kastner, D.B., Coulter, M.E., Roumis, D.K., Eden, U.T., and Frank, L.M. (2021). Hippocampal replay reflects specific past experiences rather than a plan for subsequent choice. Neuron _109_ , 3149–3163. 10.1016/j.neuron.2021.07.029. 

98. Stark, E., Levi, A., and Rotstein, H.G. (2022). Network resonance can be generated independently at distinct levels of neuronal organization 10.1371/journal.pcbi.1010364. 

99. Dempster, A. P., Laird, N. M., and Rubin, D. B (1977). Maximum likelihood from incomplete data via the em algorithm. Journal of the royal statistical society: series B `–` 

(methodological), _39_ (1):1 22. 

100. Pillow, J. W., Shlens, J., Paninski, L., Sher, A., Litke, A. M., Chichilnisky, E. J., and Simoncelli, E. P. (2008). Spatio-temporal correlations and visual signalling in a complete `–` 

neuronal population. Nature, _454_ (7207):995 999. 

