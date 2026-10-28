---
license: cc-by-4.0
language:
- aae
- aal
- aao
- abn
- abr
- abs
- abv
- acm
- acw
- acx
- adf
- aeb
- aec
- afb
- afo
- ahl
- ahs
- aju
- ala
- aln
- alo
- amu
- anc
- ank
- anp
- anw
- aom
- apc
- apd
- arq
- ars
- ary
- arz
- avl
- awo
- ayl
- ayp
- bbu
- bcs
- bcy
- bda
- bde
- bdm
- bew
- bhb
- bhh
- bho
- bhp
- bhr
- bjj
- bjk
- bjn
- bjt
- bky
- bmm
- bmq
- bns
- bou
- bqg
- bra
- brh
- brx
- bsj
- btm
- bug
- buo
- bux
- bwr
- bxf
- byc
- bys
- byx
- bzc
- bzw
- ccg
- cen
- cfa
- cgg
- chq
- ckl
- ckr
- cky
- cte
- ctl
- dbd
- dcc
- deg
- dgh
- dje
- dty
- dzg
- ebu
- ego
- eiv
- ekr
- elm
- ets
- etu
- ext
- eyo
- fat
- ffm
- fia
- fip
- fkk
- fuc
- fue
- fuf
- fuh
- fui
- fuq
- fuv
- gbm
- gbr
- gby
- gcc
- gdf
- ges
- gjk
- glw
- gol
- gom
- gsl
- gui
- gur
- guz
- gwe
- gyz
- hah
- hao
- haw
- hbb
- hz
- hia
- hkk
- hla
- hno
- hoj
- hue
- hul
- hwo
- ida
- idu
- ijc
- ijn
- ikw
- ish
- iso
- its
- itw
- itz
- jal
- jax
- jmx
- jns
- juk
- juo
- kai
- kaj
- ks
- kbl
- kbt
- kcq
- keu
- kfe
- kfk
- kfp
- kjc
- kjk
- kmy
- kna
- knn
- kol
- koo
- kpo
- kqo
- ksd
- kto
- kj
- kuh
- kwm
- kxp
- kyx
- lag
- lcm
- ldb
- lij
- lir
- lkb
- lla
- lnu
- loa
- lto
- lus
- lwg
- mab
- maf
- max
- mde
- mek
- mer
- meu
- mfm
- mfn
- mfo
- mfv
- mgi
- mig
- miu
- mkf
- mlq
- mne
- mqy
- mrr
- mrt
- msh
- msw
- mtr
- mtu
- mtx
- mui
- mxs
- mxy
- mzl
- nal
- nap
- nbh
- ncf
- nco
- ndi
- ng
- ngi
- nhg
- nhn
- nhq
- nja
- noe
- odk
- odu
- ogo
- orc
- pbs
- pbt
- pbu
- pex
- phr
- pip
- piy
- pko
- plt
- pmq
- pms
- pmy
- pnb
- poc
- poe
- pow
- pst
- qug
- qum
- quv
- rag
- rob
- rof
- roo
- rth
- sau
- say
- scn
- shu
- si
- sip
- siw
- sjr
- skg
- snc
- snk
- sol
- sps
- src
- sro
- ste
- sua
- tan
- tbf
- tcf
- tcy
- tdn
- tdx
- tgc
- the
- thq
- thr
- thv
- tio
- tkg
- tkt
- tlp
- tpl
- tpz
- tqp
- trp
- trq
- ttj
- ttr
- ttu
- tul
- tuq
- tuv
- tuy
- tvo
- twu
- txs
- txy
- uki
- uzn
- vai
- ver
- vmc
- vmj
- vmm
- vmp
- vmz
- vro
- wci
- weo
- wja
- wji
- wof
- xmv
- xmw
- xpe
- xti
- xtu
- yay
- ydd
- yer
- yes
- zga
- zoh
- zor
- zpv
- zpy
- ztg
- ztn
- ztp
- zts
- ztu
configs:
- config_name: default
  data_files:
  - split: train
    path: data/*/train-*
  - split: dev
    path: data/*/dev-*
  - split: test
    path: data/*/test-*
- config_name: aae_Latn
  data_files:
  - split: train
    path: data/aae_Latn/train-*
- config_name: aal_Latn
  data_files:
  - split: train
    path: data/aal_Latn/train-*
  - split: dev
    path: data/aal_Latn/dev-*
  - split: test
    path: data/aal_Latn/test-*
- config_name: aao_Arab
  data_files:
  - split: train
    path: data/aao_Arab/train-*
- config_name: abn_Latn
  data_files:
  - split: dev
    path: data/abn_Latn/dev-*
  - split: train
    path: data/abn_Latn/train-*
  - split: test
    path: data/abn_Latn/test-*
- config_name: abr_Latn
  data_files:
  - split: test
    path: data/abr_Latn/test-*
  - split: train
    path: data/abr_Latn/train-*
  - split: dev
    path: data/abr_Latn/dev-*
- config_name: abs_Latn
  data_files:
  - split: train
    path: data/abs_Latn/train-*
  - split: test
    path: data/abs_Latn/test-*
  - split: dev
    path: data/abs_Latn/dev-*
- config_name: abv_Arab
  data_files:
  - split: train
    path: data/abv_Arab/train-*
- config_name: acm_Arab
  data_files:
  - split: train
    path: data/acm_Arab/train-*
- config_name: acw_Arab
  data_files:
  - split: train
    path: data/acw_Arab/train-*
- config_name: acx_Arab
  data_files:
  - split: train
    path: data/acx_Arab/train-*
- config_name: adf_Arab
  data_files:
  - split: train
    path: data/adf_Arab/train-*
- config_name: aeb_Arab
  data_files:
  - split: train
    path: data/aeb_Arab/train-*
- config_name: aec_Arab
  data_files:
  - split: train
    path: data/aec_Arab/train-*
  - split: dev
    path: data/aec_Arab/dev-*
  - split: test
    path: data/aec_Arab/test-*
- config_name: afb_Arab
  data_files:
  - split: train
    path: data/afb_Arab/train-*
- config_name: afo_Latn
  data_files:
  - split: test
    path: data/afo_Latn/test-*
  - split: train
    path: data/afo_Latn/train-*
  - split: dev
    path: data/afo_Latn/dev-*
- config_name: ahl_Latn
  data_files:
  - split: train
    path: data/ahl_Latn/train-*
  - split: test
    path: data/ahl_Latn/test-*
  - split: dev
    path: data/ahl_Latn/dev-*
- config_name: ahs_Latn
  data_files:
  - split: dev
    path: data/ahs_Latn/dev-*
  - split: train
    path: data/ahs_Latn/train-*
  - split: test
    path: data/ahs_Latn/test-*
- config_name: aju_Hebr
  data_files:
  - split: train
    path: data/aju_Hebr/train-*
- config_name: ala_Latn
  data_files:
  - split: train
    path: data/ala_Latn/train-*
  - split: test
    path: data/ala_Latn/test-*
  - split: dev
    path: data/ala_Latn/dev-*
- config_name: aln_Latn
  data_files:
  - split: train
    path: data/aln_Latn/train-*
- config_name: alo_Latn
  data_files:
  - split: train
    path: data/alo_Latn/train-*
  - split: test
    path: data/alo_Latn/test-*
  - split: dev
    path: data/alo_Latn/dev-*
- config_name: amu_Latn
  data_files:
  - split: train
    path: data/amu_Latn/train-*
  - split: test
    path: data/amu_Latn/test-*
  - split: dev
    path: data/amu_Latn/dev-*
- config_name: anc_Latn
  data_files:
  - split: train
    path: data/anc_Latn/train-*
  - split: test
    path: data/anc_Latn/test-*
  - split: dev
    path: data/anc_Latn/dev-*
- config_name: ank_Latn
  data_files:
  - split: dev
    path: data/ank_Latn/dev-*
  - split: test
    path: data/ank_Latn/test-*
  - split: train
    path: data/ank_Latn/train-*
- config_name: anp_Deva
  data_files:
  - split: test
    path: data/anp_Deva/test-*
  - split: dev
    path: data/anp_Deva/dev-*
  - split: train
    path: data/anp_Deva/train-*
- config_name: anw_Latn
  data_files:
  - split: train
    path: data/anw_Latn/train-*
  - split: dev
    path: data/anw_Latn/dev-*
  - split: test
    path: data/anw_Latn/test-*
- config_name: aom_Latn
  data_files:
  - split: train
    path: data/aom_Latn/train-*
  - split: dev
    path: data/aom_Latn/dev-*
  - split: test
    path: data/aom_Latn/test-*
- config_name: apc_Arab
  data_files:
  - split: train
    path: data/apc_Arab/train-*
  - split: dev
    path: data/apc_Arab/dev-*
  - split: test
    path: data/apc_Arab/test-*
- config_name: apd_Arab
  data_files:
  - split: train
    path: data/apd_Arab/train-*
  - split: dev
    path: data/apd_Arab/dev-*
  - split: test
    path: data/apd_Arab/test-*
- config_name: arq_Arab
  data_files:
  - split: train
    path: data/arq_Arab/train-*
- config_name: ars_Arab
  data_files:
  - split: train
    path: data/ars_Arab/train-*
- config_name: ary_Arab
  data_files:
  - split: test
    path: data/ary_Arab/test-*
  - split: train
    path: data/ary_Arab/train-*
- config_name: arz_Arab
  data_files:
  - split: dev
    path: data/arz_Arab/dev-*
  - split: train
    path: data/arz_Arab/train-*
  - split: test
    path: data/arz_Arab/test-*
- config_name: avl_Arab
  data_files:
  - split: train
    path: data/avl_Arab/train-*
- config_name: awo_Latn
  data_files:
  - split: train
    path: data/awo_Latn/train-*
  - split: dev
    path: data/awo_Latn/dev-*
  - split: test
    path: data/awo_Latn/test-*
- config_name: ayl_Arab
  data_files:
  - split: train
    path: data/ayl_Arab/train-*
  - split: dev
    path: data/ayl_Arab/dev-*
  - split: test
    path: data/ayl_Arab/test-*
- config_name: ayp_Arab
  data_files:
  - split: train
    path: data/ayp_Arab/train-*
  - split: dev
    path: data/ayp_Arab/dev-*
  - split: test
    path: data/ayp_Arab/test-*
- config_name: bbu_Latn
  data_files:
  - split: dev
    path: data/bbu_Latn/dev-*
  - split: test
    path: data/bbu_Latn/test-*
  - split: train
    path: data/bbu_Latn/train-*
- config_name: bcs_Latn
  data_files:
  - split: train
    path: data/bcs_Latn/train-*
  - split: test
    path: data/bcs_Latn/test-*
  - split: dev
    path: data/bcs_Latn/dev-*
- config_name: bcy_Latn
  data_files:
  - split: train
    path: data/bcy_Latn/train-*
  - split: test
    path: data/bcy_Latn/test-*
  - split: dev
    path: data/bcy_Latn/dev-*
- config_name: bda_Latn
  data_files:
  - split: train
    path: data/bda_Latn/train-*
  - split: dev
    path: data/bda_Latn/dev-*
  - split: test
    path: data/bda_Latn/test-*
- config_name: bde_Latn
  data_files:
  - split: train
    path: data/bde_Latn/train-*
  - split: dev
    path: data/bde_Latn/dev-*
  - split: test
    path: data/bde_Latn/test-*
- config_name: bdm_Latn
  data_files:
  - split: train
    path: data/bdm_Latn/train-*
  - split: dev
    path: data/bdm_Latn/dev-*
  - split: test
    path: data/bdm_Latn/test-*
- config_name: bew_Latn
  data_files:
  - split: dev
    path: data/bew_Latn/dev-*
  - split: train
    path: data/bew_Latn/train-*
  - split: test
    path: data/bew_Latn/test-*
- config_name: bhb_Deva
  data_files:
  - split: train
    path: data/bhb_Deva/train-*
  - split: dev
    path: data/bhb_Deva/dev-*
- config_name: bhh_Cyrl
  data_files:
  - split: train
    path: data/bhh_Cyrl/train-*
  - split: dev
    path: data/bhh_Cyrl/dev-*
  - split: test
    path: data/bhh_Cyrl/test-*
- config_name: bho_Deva
  data_files:
  - split: test
    path: data/bho_Deva/test-*
  - split: train
    path: data/bho_Deva/train-*
  - split: dev
    path: data/bho_Deva/dev-*
- config_name: bhp_Latn
  data_files:
  - split: train
    path: data/bhp_Latn/train-*
  - split: dev
    path: data/bhp_Latn/dev-*
  - split: test
    path: data/bhp_Latn/test-*
- config_name: bhr_Latn
  data_files:
  - split: train
    path: data/bhr_Latn/train-*
- config_name: bjj_Deva
  data_files:
  - split: train
    path: data/bjj_Deva/train-*
  - split: test
    path: data/bjj_Deva/test-*
  - split: dev
    path: data/bjj_Deva/dev-*
- config_name: bjk_Latn
  data_files:
  - split: test
    path: data/bjk_Latn/test-*
  - split: train
    path: data/bjk_Latn/train-*
  - split: dev
    path: data/bjk_Latn/dev-*
- config_name: bjn_Latn
  data_files:
  - split: train
    path: data/bjn_Latn/train-*
  - split: test
    path: data/bjn_Latn/test-*
  - split: dev
    path: data/bjn_Latn/dev-*
- config_name: bjt_Latn
  data_files:
  - split: test
    path: data/bjt_Latn/test-*
  - split: train
    path: data/bjt_Latn/train-*
  - split: dev
    path: data/bjt_Latn/dev-*
- config_name: bky_Latn
  data_files:
  - split: train
    path: data/bky_Latn/train-*
  - split: test
    path: data/bky_Latn/test-*
  - split: dev
    path: data/bky_Latn/dev-*
- config_name: bmm_Latn
  data_files:
  - split: train
    path: data/bmm_Latn/train-*
- config_name: bmq_Latn
  data_files:
  - split: train
    path: data/bmq_Latn/train-*
- config_name: bns_Deva
  data_files:
  - split: train
    path: data/bns_Deva/train-*
  - split: test
    path: data/bns_Deva/test-*
  - split: dev
    path: data/bns_Deva/dev-*
- config_name: bou_Latn
  data_files:
  - split: train
    path: data/bou_Latn/train-*
  - split: dev
    path: data/bou_Latn/dev-*
  - split: test
    path: data/bou_Latn/test-*
- config_name: bqg_Latn
  data_files:
  - split: train
    path: data/bqg_Latn/train-*
  - split: dev
    path: data/bqg_Latn/dev-*
  - split: test
    path: data/bqg_Latn/test-*
- config_name: bra_Deva
  data_files:
  - split: train
    path: data/bra_Deva/train-*
  - split: dev
    path: data/bra_Deva/dev-*
  - split: test
    path: data/bra_Deva/test-*
- config_name: brh_Arab
  data_files:
  - split: train
    path: data/brh_Arab/train-*
  - split: test
    path: data/brh_Arab/test-*
  - split: dev
    path: data/brh_Arab/dev-*
- config_name: brx_Deva
  data_files:
  - split: train
    path: data/brx_Deva/train-*
  - split: dev
    path: data/brx_Deva/dev-*
  - split: test
    path: data/brx_Deva/test-*
- config_name: bsj_Latn
  data_files:
  - split: dev
    path: data/bsj_Latn/dev-*
  - split: test
    path: data/bsj_Latn/test-*
  - split: train
    path: data/bsj_Latn/train-*
- config_name: btm_Latn
  data_files:
  - split: train
    path: data/btm_Latn/train-*
  - split: dev
    path: data/btm_Latn/dev-*
  - split: test
    path: data/btm_Latn/test-*
- config_name: bug_Latn
  data_files:
  - split: train
    path: data/bug_Latn/train-*
  - split: dev
    path: data/bug_Latn/dev-*
  - split: test
    path: data/bug_Latn/test-*
- config_name: buo_Latn
  data_files:
  - split: train
    path: data/buo_Latn/train-*
  - split: dev
    path: data/buo_Latn/dev-*
  - split: test
    path: data/buo_Latn/test-*
- config_name: bux_Latn
  data_files:
  - split: train
    path: data/bux_Latn/train-*
  - split: test
    path: data/bux_Latn/test-*
  - split: dev
    path: data/bux_Latn/dev-*
- config_name: bwr_Latn
  data_files:
  - split: train
    path: data/bwr_Latn/train-*
  - split: dev
    path: data/bwr_Latn/dev-*
  - split: test
    path: data/bwr_Latn/test-*
- config_name: bxf_Latn
  data_files:
  - split: train
    path: data/bxf_Latn/train-*
  - split: test
    path: data/bxf_Latn/test-*
  - split: dev
    path: data/bxf_Latn/dev-*
- config_name: byc_Latn
  data_files:
  - split: train
    path: data/byc_Latn/train-*
  - split: dev
    path: data/byc_Latn/dev-*
  - split: test
    path: data/byc_Latn/test-*
- config_name: bys_Latn
  data_files:
  - split: dev
    path: data/bys_Latn/dev-*
  - split: train
    path: data/bys_Latn/train-*
  - split: test
    path: data/bys_Latn/test-*
- config_name: byx_Latn
  data_files:
  - split: dev
    path: data/byx_Latn/dev-*
  - split: train
    path: data/byx_Latn/train-*
  - split: test
    path: data/byx_Latn/test-*
- config_name: bzc_Latn
  data_files:
  - split: train
    path: data/bzc_Latn/train-*
- config_name: bzw_Latn
  data_files:
  - split: train
    path: data/bzw_Latn/train-*
  - split: test
    path: data/bzw_Latn/test-*
  - split: dev
    path: data/bzw_Latn/dev-*
- config_name: ccg_Latn
  data_files:
  - split: train
    path: data/ccg_Latn/train-*
  - split: dev
    path: data/ccg_Latn/dev-*
  - split: test
    path: data/ccg_Latn/test-*
- config_name: cen_Latn
  data_files:
  - split: train
    path: data/cen_Latn/train-*
  - split: test
    path: data/cen_Latn/test-*
  - split: dev
    path: data/cen_Latn/dev-*
- config_name: cfa_Latn
  data_files:
  - split: train
    path: data/cfa_Latn/train-*
  - split: test
    path: data/cfa_Latn/test-*
  - split: dev
    path: data/cfa_Latn/dev-*
- config_name: cgg_Latn
  data_files:
  - split: train
    path: data/cgg_Latn/train-*
  - split: dev
    path: data/cgg_Latn/dev-*
  - split: test
    path: data/cgg_Latn/test-*
- config_name: chq_Latn
  data_files:
  - split: train
    path: data/chq_Latn/train-*
  - split: dev
    path: data/chq_Latn/dev-*
  - split: test
    path: data/chq_Latn/test-*
- config_name: ckl_Latn
  data_files:
  - split: dev
    path: data/ckl_Latn/dev-*
  - split: train
    path: data/ckl_Latn/train-*
  - split: test
    path: data/ckl_Latn/test-*
- config_name: ckr_Latn
  data_files:
  - split: test
    path: data/ckr_Latn/test-*
  - split: dev
    path: data/ckr_Latn/dev-*
  - split: train
    path: data/ckr_Latn/train-*
- config_name: cky_Latn
  data_files:
  - split: test
    path: data/cky_Latn/test-*
  - split: dev
    path: data/cky_Latn/dev-*
  - split: train
    path: data/cky_Latn/train-*
- config_name: cte_Latn
  data_files:
  - split: test
    path: data/cte_Latn/test-*
  - split: dev
    path: data/cte_Latn/dev-*
  - split: train
    path: data/cte_Latn/train-*
- config_name: ctl_Latn
  data_files:
  - split: train
    path: data/ctl_Latn/train-*
  - split: test
    path: data/ctl_Latn/test-*
  - split: dev
    path: data/ctl_Latn/dev-*
- config_name: dbd_Latn
  data_files:
  - split: train
    path: data/dbd_Latn/train-*
  - split: dev
    path: data/dbd_Latn/dev-*
  - split: test
    path: data/dbd_Latn/test-*
- config_name: dcc_Arab
  data_files:
  - split: dev
    path: data/dcc_Arab/dev-*
  - split: train
    path: data/dcc_Arab/train-*
  - split: test
    path: data/dcc_Arab/test-*
- config_name: deg_Latn
  data_files:
  - split: dev
    path: data/deg_Latn/dev-*
  - split: train
    path: data/deg_Latn/train-*
  - split: test
    path: data/deg_Latn/test-*
- config_name: dgh_Latn
  data_files:
  - split: train
    path: data/dgh_Latn/train-*
  - split: test
    path: data/dgh_Latn/test-*
  - split: dev
    path: data/dgh_Latn/dev-*
- config_name: dje_Latn
  data_files:
  - split: train
    path: data/dje_Latn/train-*
  - split: test
    path: data/dje_Latn/test-*
  - split: dev
    path: data/dje_Latn/dev-*
- config_name: dty_Deva
  data_files:
  - split: dev
    path: data/dty_Deva/dev-*
  - split: train
    path: data/dty_Deva/train-*
  - split: test
    path: data/dty_Deva/test-*
- config_name: dzg_Latn
  data_files:
  - split: train
    path: data/dzg_Latn/train-*
  - split: dev
    path: data/dzg_Latn/dev-*
  - split: test
    path: data/dzg_Latn/test-*
- config_name: ebu_Latn
  data_files:
  - split: train
    path: data/ebu_Latn/train-*
  - split: dev
    path: data/ebu_Latn/dev-*
  - split: test
    path: data/ebu_Latn/test-*
- config_name: ego_Latn
  data_files:
  - split: train
    path: data/ego_Latn/train-*
  - split: dev
    path: data/ego_Latn/dev-*
  - split: test
    path: data/ego_Latn/test-*
- config_name: eiv_Latn
  data_files:
  - split: dev
    path: data/eiv_Latn/dev-*
  - split: train
    path: data/eiv_Latn/train-*
  - split: test
    path: data/eiv_Latn/test-*
- config_name: ekr_Latn
  data_files:
  - split: train
    path: data/ekr_Latn/train-*
  - split: dev
    path: data/ekr_Latn/dev-*
  - split: test
    path: data/ekr_Latn/test-*
- config_name: elm_Latn
  data_files:
  - split: test
    path: data/elm_Latn/test-*
  - split: train
    path: data/elm_Latn/train-*
  - split: dev
    path: data/elm_Latn/dev-*
- config_name: ets_Latn
  data_files:
  - split: train
    path: data/ets_Latn/train-*
  - split: dev
    path: data/ets_Latn/dev-*
  - split: test
    path: data/ets_Latn/test-*
- config_name: etu_Latn
  data_files:
  - split: train
    path: data/etu_Latn/train-*
  - split: dev
    path: data/etu_Latn/dev-*
  - split: test
    path: data/etu_Latn/test-*
- config_name: ext_Latn
  data_files:
  - split: train
    path: data/ext_Latn/train-*
- config_name: eyo_Latn
  data_files:
  - split: train
    path: data/eyo_Latn/train-*
  - split: dev
    path: data/eyo_Latn/dev-*
  - split: test
    path: data/eyo_Latn/test-*
- config_name: fat_Latn
  data_files:
  - split: train
    path: data/fat_Latn/train-*
- config_name: ffm_Latn
  data_files:
  - split: train
    path: data/ffm_Latn/train-*
- config_name: fia_Latn
  data_files:
  - split: dev
    path: data/fia_Latn/dev-*
  - split: train
    path: data/fia_Latn/train-*
  - split: test
    path: data/fia_Latn/test-*
- config_name: fip_Latn
  data_files:
  - split: dev
    path: data/fip_Latn/dev-*
  - split: train
    path: data/fip_Latn/train-*
  - split: test
    path: data/fip_Latn/test-*
- config_name: fkk_Latn
  data_files:
  - split: train
    path: data/fkk_Latn/train-*
  - split: dev
    path: data/fkk_Latn/dev-*
  - split: test
    path: data/fkk_Latn/test-*
- config_name: fuc_Latn
  data_files:
  - split: train
    path: data/fuc_Latn/train-*
- config_name: fue_Latn
  data_files:
  - split: train
    path: data/fue_Latn/train-*
- config_name: fuf_Latn
  data_files:
  - split: train
    path: data/fuf_Latn/train-*
- config_name: fuh_Latn
  data_files:
  - split: train
    path: data/fuh_Latn/train-*
- config_name: fui_Latn
  data_files:
  - split: train
    path: data/fui_Latn/train-*
- config_name: fuq_Latn
  data_files:
  - split: test
    path: data/fuq_Latn/test-*
  - split: dev
    path: data/fuq_Latn/dev-*
  - split: train
    path: data/fuq_Latn/train-*
- config_name: fuv_Latn
  data_files:
  - split: train
    path: data/fuv_Latn/train-*
  - split: test
    path: data/fuv_Latn/test-*
  - split: dev
    path: data/fuv_Latn/dev-*
- config_name: gbm_Deva
  data_files:
  - split: dev
    path: data/gbm_Deva/dev-*
  - split: train
    path: data/gbm_Deva/train-*
  - split: test
    path: data/gbm_Deva/test-*
- config_name: gbr_Latn
  data_files:
  - split: dev
    path: data/gbr_Latn/dev-*
  - split: train
    path: data/gbr_Latn/train-*
  - split: test
    path: data/gbr_Latn/test-*
- config_name: gby_Latn
  data_files:
  - split: train
    path: data/gby_Latn/train-*
  - split: dev
    path: data/gby_Latn/dev-*
  - split: test
    path: data/gby_Latn/test-*
- config_name: gcc_Latn
  data_files:
  - split: train
    path: data/gcc_Latn/train-*
  - split: test
    path: data/gcc_Latn/test-*
  - split: dev
    path: data/gcc_Latn/dev-*
- config_name: gdf_Latn
  data_files:
  - split: train
    path: data/gdf_Latn/train-*
  - split: dev
    path: data/gdf_Latn/dev-*
  - split: test
    path: data/gdf_Latn/test-*
- config_name: ges_Latn
  data_files:
  - split: train
    path: data/ges_Latn/train-*
  - split: test
    path: data/ges_Latn/test-*
  - split: dev
    path: data/ges_Latn/dev-*
- config_name: gjk_Arab
  data_files:
  - split: train
    path: data/gjk_Arab/train-*
  - split: dev
    path: data/gjk_Arab/dev-*
  - split: test
    path: data/gjk_Arab/test-*
- config_name: glw_Latn
  data_files:
  - split: dev
    path: data/glw_Latn/dev-*
  - split: train
    path: data/glw_Latn/train-*
  - split: test
    path: data/glw_Latn/test-*
- config_name: gol_Latn
  data_files:
  - split: train
    path: data/gol_Latn/train-*
  - split: dev
    path: data/gol_Latn/dev-*
  - split: test
    path: data/gol_Latn/test-*
- config_name: gom_Deva
  data_files:
  - split: train
    path: data/gom_Deva/train-*
  - split: dev
    path: data/gom_Deva/dev-*
  - split: test
    path: data/gom_Deva/test-*
- config_name: gsl_Latn
  data_files:
  - split: train
    path: data/gsl_Latn/train-*
  - split: test
    path: data/gsl_Latn/test-*
  - split: dev
    path: data/gsl_Latn/dev-*
- config_name: gui_Latn
  data_files:
  - split: train
    path: data/gui_Latn/train-*
- config_name: gur_Latn
  data_files:
  - split: train
    path: data/gur_Latn/train-*
  - split: dev
    path: data/gur_Latn/dev-*
  - split: test
    path: data/gur_Latn/test-*
- config_name: guz_Latn
  data_files:
  - split: train
    path: data/guz_Latn/train-*
  - split: dev
    path: data/guz_Latn/dev-*
  - split: test
    path: data/guz_Latn/test-*
- config_name: gwe_Latn
  data_files:
  - split: train
    path: data/gwe_Latn/train-*
  - split: test
    path: data/gwe_Latn/test-*
  - split: dev
    path: data/gwe_Latn/dev-*
- config_name: gyz_Latn
  data_files:
  - split: train
    path: data/gyz_Latn/train-*
  - split: test
    path: data/gyz_Latn/test-*
  - split: dev
    path: data/gyz_Latn/dev-*
- config_name: hah_Latn
  data_files:
  - split: train
    path: data/hah_Latn/train-*
  - split: dev
    path: data/hah_Latn/dev-*
  - split: test
    path: data/hah_Latn/test-*
- config_name: hao_Latn
  data_files:
  - split: dev
    path: data/hao_Latn/dev-*
  - split: train
    path: data/hao_Latn/train-*
  - split: test
    path: data/hao_Latn/test-*
- config_name: haw_Latn
  data_files:
  - split: train
    path: data/haw_Latn/train-*
  - split: dev
    path: data/haw_Latn/dev-*
  - split: test
    path: data/haw_Latn/test-*
- config_name: hbb_Latn
  data_files:
  - split: train
    path: data/hbb_Latn/train-*
  - split: test
    path: data/hbb_Latn/test-*
  - split: dev
    path: data/hbb_Latn/dev-*
- config_name: her_Latn
  data_files:
  - split: dev
    path: data/her_Latn/dev-*
  - split: train
    path: data/her_Latn/train-*
  - split: test
    path: data/her_Latn/test-*
- config_name: hia_Latn
  data_files:
  - split: train
    path: data/hia_Latn/train-*
  - split: dev
    path: data/hia_Latn/dev-*
  - split: test
    path: data/hia_Latn/test-*
- config_name: hkk_Latn
  data_files:
  - split: dev
    path: data/hkk_Latn/dev-*
  - split: test
    path: data/hkk_Latn/test-*
  - split: train
    path: data/hkk_Latn/train-*
- config_name: hla_Latn
  data_files:
  - split: test
    path: data/hla_Latn/test-*
  - split: train
    path: data/hla_Latn/train-*
  - split: dev
    path: data/hla_Latn/dev-*
- config_name: hno_Arab
  data_files:
  - split: train
    path: data/hno_Arab/train-*
  - split: test
    path: data/hno_Arab/test-*
  - split: dev
    path: data/hno_Arab/dev-*
- config_name: hoj_Deva
  data_files:
  - split: train
    path: data/hoj_Deva/train-*
- config_name: hue_Latn
  data_files:
  - split: dev
    path: data/hue_Latn/dev-*
  - split: train
    path: data/hue_Latn/train-*
  - split: test
    path: data/hue_Latn/test-*
- config_name: hul_Latn
  data_files:
  - split: train
    path: data/hul_Latn/train-*
  - split: dev
    path: data/hul_Latn/dev-*
  - split: test
    path: data/hul_Latn/test-*
- config_name: hwo_Latn
  data_files:
  - split: train
    path: data/hwo_Latn/train-*
  - split: test
    path: data/hwo_Latn/test-*
  - split: dev
    path: data/hwo_Latn/dev-*
- config_name: ida_Latn
  data_files:
  - split: train
    path: data/ida_Latn/train-*
  - split: dev
    path: data/ida_Latn/dev-*
  - split: test
    path: data/ida_Latn/test-*
- config_name: idu_Latn
  data_files:
  - split: train
    path: data/idu_Latn/train-*
  - split: test
    path: data/idu_Latn/test-*
  - split: dev
    path: data/idu_Latn/dev-*
- config_name: ijc_Latn
  data_files:
  - split: dev
    path: data/ijc_Latn/dev-*
  - split: train
    path: data/ijc_Latn/train-*
  - split: test
    path: data/ijc_Latn/test-*
- config_name: ijn_Latn
  data_files:
  - split: train
    path: data/ijn_Latn/train-*
  - split: dev
    path: data/ijn_Latn/dev-*
  - split: test
    path: data/ijn_Latn/test-*
- config_name: ikw_Latn
  data_files:
  - split: train
    path: data/ikw_Latn/train-*
  - split: dev
    path: data/ikw_Latn/dev-*
  - split: test
    path: data/ikw_Latn/test-*
- config_name: ish_Latn
  data_files:
  - split: train
    path: data/ish_Latn/train-*
  - split: test
    path: data/ish_Latn/test-*
  - split: dev
    path: data/ish_Latn/dev-*
- config_name: iso_Latn
  data_files:
  - split: dev
    path: data/iso_Latn/dev-*
  - split: train
    path: data/iso_Latn/train-*
  - split: test
    path: data/iso_Latn/test-*
- config_name: its_Latn
  data_files:
  - split: train
    path: data/its_Latn/train-*
  - split: dev
    path: data/its_Latn/dev-*
  - split: test
    path: data/its_Latn/test-*
- config_name: itw_Latn
  data_files:
  - split: train
    path: data/itw_Latn/train-*
  - split: test
    path: data/itw_Latn/test-*
  - split: dev
    path: data/itw_Latn/dev-*
- config_name: itz_Latn
  data_files:
  - split: train
    path: data/itz_Latn/train-*
  - split: dev
    path: data/itz_Latn/dev-*
  - split: test
    path: data/itz_Latn/test-*
- config_name: jal_Latn
  data_files:
  - split: train
    path: data/jal_Latn/train-*
  - split: test
    path: data/jal_Latn/test-*
  - split: dev
    path: data/jal_Latn/dev-*
- config_name: jax_Latn
  data_files:
  - split: train
    path: data/jax_Latn/train-*
  - split: dev
    path: data/jax_Latn/dev-*
  - split: test
    path: data/jax_Latn/test-*
- config_name: jmx_Latn
  data_files:
  - split: train
    path: data/jmx_Latn/train-*
  - split: test
    path: data/jmx_Latn/test-*
  - split: dev
    path: data/jmx_Latn/dev-*
- config_name: jns_Deva
  data_files:
  - split: train
    path: data/jns_Deva/train-*
- config_name: juk_Latn
  data_files:
  - split: train
    path: data/juk_Latn/train-*
  - split: dev
    path: data/juk_Latn/dev-*
  - split: test
    path: data/juk_Latn/test-*
- config_name: juo_Latn
  data_files:
  - split: train
    path: data/juo_Latn/train-*
  - split: test
    path: data/juo_Latn/test-*
  - split: dev
    path: data/juo_Latn/dev-*
- config_name: kai_Latn
  data_files:
  - split: train
    path: data/kai_Latn/train-*
  - split: test
    path: data/kai_Latn/test-*
  - split: dev
    path: data/kai_Latn/dev-*
- config_name: kaj_Latn
  data_files:
  - split: train
    path: data/kaj_Latn/train-*
  - split: test
    path: data/kaj_Latn/test-*
  - split: dev
    path: data/kaj_Latn/dev-*
- config_name: kas_Arab
  data_files:
  - split: train
    path: data/kas_Arab/train-*
  - split: dev
    path: data/kas_Arab/dev-*
  - split: test
    path: data/kas_Arab/test-*
- config_name: kbl_Latn
  data_files:
  - split: train
    path: data/kbl_Latn/train-*
  - split: dev
    path: data/kbl_Latn/dev-*
  - split: test
    path: data/kbl_Latn/test-*
- config_name: kbt_Latn
  data_files:
  - split: test
    path: data/kbt_Latn/test-*
  - split: dev
    path: data/kbt_Latn/dev-*
  - split: train
    path: data/kbt_Latn/train-*
- config_name: kcq_Latn
  data_files:
  - split: train
    path: data/kcq_Latn/train-*
  - split: test
    path: data/kcq_Latn/test-*
  - split: dev
    path: data/kcq_Latn/dev-*
- config_name: keu_Latn
  data_files:
  - split: train
    path: data/keu_Latn/train-*
  - split: dev
    path: data/keu_Latn/dev-*
  - split: test
    path: data/keu_Latn/test-*
- config_name: kfe_Taml
  data_files:
  - split: train
    path: data/kfe_Taml/train-*
- config_name: kfk_Deva
  data_files:
  - split: dev
    path: data/kfk_Deva/dev-*
  - split: test
    path: data/kfk_Deva/test-*
  - split: train
    path: data/kfk_Deva/train-*
- config_name: kfp_Deva
  data_files:
  - split: train
    path: data/kfp_Deva/train-*
- config_name: kjc_Latn
  data_files:
  - split: train
    path: data/kjc_Latn/train-*
  - split: dev
    path: data/kjc_Latn/dev-*
  - split: test
    path: data/kjc_Latn/test-*
- config_name: kjk_Latn
  data_files:
  - split: train
    path: data/kjk_Latn/train-*
  - split: test
    path: data/kjk_Latn/test-*
  - split: dev
    path: data/kjk_Latn/dev-*
- config_name: kmy_Latn
  data_files:
  - split: train
    path: data/kmy_Latn/train-*
  - split: test
    path: data/kmy_Latn/test-*
  - split: dev
    path: data/kmy_Latn/dev-*
- config_name: kna_Latn
  data_files:
  - split: test
    path: data/kna_Latn/test-*
  - split: train
    path: data/kna_Latn/train-*
  - split: dev
    path: data/kna_Latn/dev-*
- config_name: knn_Deva
  data_files:
  - split: dev
    path: data/knn_Deva/dev-*
  - split: train
    path: data/knn_Deva/train-*
  - split: test
    path: data/knn_Deva/test-*
- config_name: kol_Latn
  data_files:
  - split: train
    path: data/kol_Latn/train-*
  - split: dev
    path: data/kol_Latn/dev-*
  - split: test
    path: data/kol_Latn/test-*
- config_name: koo_Latn
  data_files:
  - split: train
    path: data/koo_Latn/train-*
  - split: dev
    path: data/koo_Latn/dev-*
  - split: test
    path: data/koo_Latn/test-*
- config_name: kpo_Latn
  data_files:
  - split: train
    path: data/kpo_Latn/train-*
  - split: dev
    path: data/kpo_Latn/dev-*
  - split: test
    path: data/kpo_Latn/test-*
- config_name: kqo_Latn
  data_files:
  - split: train
    path: data/kqo_Latn/train-*
  - split: test
    path: data/kqo_Latn/test-*
  - split: dev
    path: data/kqo_Latn/dev-*
- config_name: ksd_Latn
  data_files:
  - split: train
    path: data/ksd_Latn/train-*
  - split: test
    path: data/ksd_Latn/test-*
  - split: dev
    path: data/ksd_Latn/dev-*
- config_name: kto_Latn
  data_files:
  - split: test
    path: data/kto_Latn/test-*
  - split: dev
    path: data/kto_Latn/dev-*
  - split: train
    path: data/kto_Latn/train-*
- config_name: kua_Latn
  data_files:
  - split: test
    path: data/kua_Latn/test-*
  - split: dev
    path: data/kua_Latn/dev-*
  - split: train
    path: data/kua_Latn/train-*
- config_name: kuh_Latn
  data_files:
  - split: train
    path: data/kuh_Latn/train-*
  - split: dev
    path: data/kuh_Latn/dev-*
  - split: test
    path: data/kuh_Latn/test-*
- config_name: kwm_Latn
  data_files:
  - split: dev
    path: data/kwm_Latn/dev-*
  - split: train
    path: data/kwm_Latn/train-*
  - split: test
    path: data/kwm_Latn/test-*
- config_name: kxp_Arab
  data_files:
  - split: test
    path: data/kxp_Arab/test-*
  - split: train
    path: data/kxp_Arab/train-*
  - split: dev
    path: data/kxp_Arab/dev-*
- config_name: kyx_Latn
  data_files:
  - split: train
    path: data/kyx_Latn/train-*
  - split: test
    path: data/kyx_Latn/test-*
  - split: dev
    path: data/kyx_Latn/dev-*
- config_name: lag_Latn
  data_files:
  - split: train
    path: data/lag_Latn/train-*
  - split: dev
    path: data/lag_Latn/dev-*
  - split: test
    path: data/lag_Latn/test-*
- config_name: lcm_Latn
  data_files:
  - split: dev
    path: data/lcm_Latn/dev-*
  - split: train
    path: data/lcm_Latn/train-*
  - split: test
    path: data/lcm_Latn/test-*
- config_name: ldb_Latn
  data_files:
  - split: test
    path: data/ldb_Latn/test-*
  - split: train
    path: data/ldb_Latn/train-*
  - split: dev
    path: data/ldb_Latn/dev-*
- config_name: lij_Latn
  data_files:
  - split: train
    path: data/lij_Latn/train-*
  - split: test
    path: data/lij_Latn/test-*
  - split: dev
    path: data/lij_Latn/dev-*
- config_name: lir_Latn
  data_files:
  - split: train
    path: data/lir_Latn/train-*
  - split: test
    path: data/lir_Latn/test-*
  - split: dev
    path: data/lir_Latn/dev-*
- config_name: lkb_Latn
  data_files:
  - split: train
    path: data/lkb_Latn/train-*
  - split: test
    path: data/lkb_Latn/test-*
  - split: dev
    path: data/lkb_Latn/dev-*
- config_name: lla_Latn
  data_files:
  - split: dev
    path: data/lla_Latn/dev-*
  - split: train
    path: data/lla_Latn/train-*
  - split: test
    path: data/lla_Latn/test-*
- config_name: lnu_Latn
  data_files:
  - split: train
    path: data/lnu_Latn/train-*
  - split: test
    path: data/lnu_Latn/test-*
  - split: dev
    path: data/lnu_Latn/dev-*
- config_name: loa_Latn
  data_files:
  - split: test
    path: data/loa_Latn/test-*
  - split: train
    path: data/loa_Latn/train-*
  - split: dev
    path: data/loa_Latn/dev-*
- config_name: lto_Latn
  data_files:
  - split: train
    path: data/lto_Latn/train-*
  - split: dev
    path: data/lto_Latn/dev-*
  - split: test
    path: data/lto_Latn/test-*
- config_name: lus_Latn
  data_files:
  - split: train
    path: data/lus_Latn/train-*
  - split: dev
    path: data/lus_Latn/dev-*
  - split: test
    path: data/lus_Latn/test-*
- config_name: lwg_Latn
  data_files:
  - split: dev
    path: data/lwg_Latn/dev-*
  - split: test
    path: data/lwg_Latn/test-*
  - split: train
    path: data/lwg_Latn/train-*
- config_name: mab_Latn
  data_files:
  - split: test
    path: data/mab_Latn/test-*
  - split: train
    path: data/mab_Latn/train-*
  - split: dev
    path: data/mab_Latn/dev-*
- config_name: maf_Latn
  data_files:
  - split: train
    path: data/maf_Latn/train-*
  - split: test
    path: data/maf_Latn/test-*
  - split: dev
    path: data/maf_Latn/dev-*
- config_name: max_Latn
  data_files:
  - split: train
    path: data/max_Latn/train-*
  - split: dev
    path: data/max_Latn/dev-*
  - split: test
    path: data/max_Latn/test-*
- config_name: mde_Latn
  data_files:
  - split: train
    path: data/mde_Latn/train-*
- config_name: mek_Latn
  data_files:
  - split: train
    path: data/mek_Latn/train-*
  - split: test
    path: data/mek_Latn/test-*
  - split: dev
    path: data/mek_Latn/dev-*
- config_name: mer_Latn
  data_files:
  - split: dev
    path: data/mer_Latn/dev-*
  - split: train
    path: data/mer_Latn/train-*
  - split: test
    path: data/mer_Latn/test-*
- config_name: meu_Latn
  data_files:
  - split: train
    path: data/meu_Latn/train-*
  - split: test
    path: data/meu_Latn/test-*
  - split: dev
    path: data/meu_Latn/dev-*
- config_name: mfm_Latn
  data_files:
  - split: test
    path: data/mfm_Latn/test-*
  - split: train
    path: data/mfm_Latn/train-*
  - split: dev
    path: data/mfm_Latn/dev-*
- config_name: mfn_Latn
  data_files:
  - split: train
    path: data/mfn_Latn/train-*
  - split: test
    path: data/mfn_Latn/test-*
  - split: dev
    path: data/mfn_Latn/dev-*
- config_name: mfo_Latn
  data_files:
  - split: train
    path: data/mfo_Latn/train-*
  - split: test
    path: data/mfo_Latn/test-*
  - split: dev
    path: data/mfo_Latn/dev-*
- config_name: mfv_Latn
  data_files:
  - split: train
    path: data/mfv_Latn/train-*
  - split: test
    path: data/mfv_Latn/test-*
  - split: dev
    path: data/mfv_Latn/dev-*
- config_name: mgi_Latn
  data_files:
  - split: train
    path: data/mgi_Latn/train-*
  - split: test
    path: data/mgi_Latn/test-*
  - split: dev
    path: data/mgi_Latn/dev-*
- config_name: mig_Latn
  data_files:
  - split: test
    path: data/mig_Latn/test-*
  - split: train
    path: data/mig_Latn/train-*
  - split: dev
    path: data/mig_Latn/dev-*
- config_name: miu_Latn
  data_files:
  - split: dev
    path: data/miu_Latn/dev-*
  - split: train
    path: data/miu_Latn/train-*
  - split: test
    path: data/miu_Latn/test-*
- config_name: mkf_Latn
  data_files:
  - split: test
    path: data/mkf_Latn/test-*
  - split: train
    path: data/mkf_Latn/train-*
  - split: dev
    path: data/mkf_Latn/dev-*
- config_name: mlq_Latn
  data_files:
  - split: dev
    path: data/mlq_Latn/dev-*
  - split: train
    path: data/mlq_Latn/train-*
  - split: test
    path: data/mlq_Latn/test-*
- config_name: mne_Latn
  data_files:
  - split: test
    path: data/mne_Latn/test-*
  - split: dev
    path: data/mne_Latn/dev-*
  - split: train
    path: data/mne_Latn/train-*
- config_name: mqy_Latn
  data_files:
  - split: test
    path: data/mqy_Latn/test-*
  - split: train
    path: data/mqy_Latn/train-*
  - split: dev
    path: data/mqy_Latn/dev-*
- config_name: mrr_Deva
  data_files:
  - split: test
    path: data/mrr_Deva/test-*
  - split: train
    path: data/mrr_Deva/train-*
  - split: dev
    path: data/mrr_Deva/dev-*
- config_name: mrt_Latn
  data_files:
  - split: train
    path: data/mrt_Latn/train-*
  - split: dev
    path: data/mrt_Latn/dev-*
  - split: test
    path: data/mrt_Latn/test-*
- config_name: msh_Latn
  data_files:
  - split: train
    path: data/msh_Latn/train-*
- config_name: msw_Latn
  data_files:
  - split: train
    path: data/msw_Latn/train-*
  - split: dev
    path: data/msw_Latn/dev-*
  - split: test
    path: data/msw_Latn/test-*
- config_name: mtr_Deva
  data_files:
  - split: test
    path: data/mtr_Deva/test-*
  - split: dev
    path: data/mtr_Deva/dev-*
  - split: train
    path: data/mtr_Deva/train-*
- config_name: mtu_Latn
  data_files:
  - split: dev
    path: data/mtu_Latn/dev-*
  - split: train
    path: data/mtu_Latn/train-*
  - split: test
    path: data/mtu_Latn/test-*
- config_name: mtx_Latn
  data_files:
  - split: dev
    path: data/mtx_Latn/dev-*
  - split: train
    path: data/mtx_Latn/train-*
  - split: test
    path: data/mtx_Latn/test-*
- config_name: mui_Latn
  data_files:
  - split: train
    path: data/mui_Latn/train-*
  - split: dev
    path: data/mui_Latn/dev-*
  - split: test
    path: data/mui_Latn/test-*
- config_name: mxs_Latn
  data_files:
  - split: test
    path: data/mxs_Latn/test-*
  - split: dev
    path: data/mxs_Latn/dev-*
  - split: train
    path: data/mxs_Latn/train-*
- config_name: mxy_Latn
  data_files:
  - split: train
    path: data/mxy_Latn/train-*
  - split: test
    path: data/mxy_Latn/test-*
  - split: dev
    path: data/mxy_Latn/dev-*
- config_name: mzl_Latn
  data_files:
  - split: train
    path: data/mzl_Latn/train-*
  - split: test
    path: data/mzl_Latn/test-*
  - split: dev
    path: data/mzl_Latn/dev-*
- config_name: nal_Latn
  data_files:
  - split: train
    path: data/nal_Latn/train-*
  - split: test
    path: data/nal_Latn/test-*
  - split: dev
    path: data/nal_Latn/dev-*
- config_name: nap_Latn
  data_files:
  - split: test
    path: data/nap_Latn/test-*
  - split: train
    path: data/nap_Latn/train-*
  - split: dev
    path: data/nap_Latn/dev-*
- config_name: nbh_Latn
  data_files:
  - split: train
    path: data/nbh_Latn/train-*
  - split: test
    path: data/nbh_Latn/test-*
  - split: dev
    path: data/nbh_Latn/dev-*
- config_name: ncf_Latn
  data_files:
  - split: dev
    path: data/ncf_Latn/dev-*
  - split: train
    path: data/ncf_Latn/train-*
  - split: test
    path: data/ncf_Latn/test-*
- config_name: nco_Latn
  data_files:
  - split: train
    path: data/nco_Latn/train-*
  - split: dev
    path: data/nco_Latn/dev-*
  - split: test
    path: data/nco_Latn/test-*
- config_name: ndi_Latn
  data_files:
  - split: train
    path: data/ndi_Latn/train-*
  - split: dev
    path: data/ndi_Latn/dev-*
  - split: test
    path: data/ndi_Latn/test-*
- config_name: ndo_Latn
  data_files:
  - split: test
    path: data/ndo_Latn/test-*
  - split: train
    path: data/ndo_Latn/train-*
  - split: dev
    path: data/ndo_Latn/dev-*
- config_name: ngi_Latn
  data_files:
  - split: dev
    path: data/ngi_Latn/dev-*
  - split: train
    path: data/ngi_Latn/train-*
  - split: test
    path: data/ngi_Latn/test-*
- config_name: nhg_Latn
  data_files:
  - split: test
    path: data/nhg_Latn/test-*
  - split: train
    path: data/nhg_Latn/train-*
  - split: dev
    path: data/nhg_Latn/dev-*
- config_name: nhn_Latn
  data_files:
  - split: train
    path: data/nhn_Latn/train-*
  - split: dev
    path: data/nhn_Latn/dev-*
  - split: test
    path: data/nhn_Latn/test-*
- config_name: nhq_Latn
  data_files:
  - split: dev
    path: data/nhq_Latn/dev-*
  - split: test
    path: data/nhq_Latn/test-*
  - split: train
    path: data/nhq_Latn/train-*
- config_name: nja_Latn
  data_files:
  - split: train
    path: data/nja_Latn/train-*
  - split: dev
    path: data/nja_Latn/dev-*
  - split: test
    path: data/nja_Latn/test-*
- config_name: noe_Deva
  data_files:
  - split: train
    path: data/noe_Deva/train-*
  - split: dev
    path: data/noe_Deva/dev-*
  - split: test
    path: data/noe_Deva/test-*
- config_name: odk_Arab
  data_files:
  - split: dev
    path: data/odk_Arab/dev-*
  - split: test
    path: data/odk_Arab/test-*
  - split: train
    path: data/odk_Arab/train-*
- config_name: odu_Latn
  data_files:
  - split: dev
    path: data/odu_Latn/dev-*
  - split: train
    path: data/odu_Latn/train-*
  - split: test
    path: data/odu_Latn/test-*
- config_name: ogo_Latn
  data_files:
  - split: test
    path: data/ogo_Latn/test-*
  - split: dev
    path: data/ogo_Latn/dev-*
  - split: train
    path: data/ogo_Latn/train-*
- config_name: orc_Latn
  data_files:
  - split: train
    path: data/orc_Latn/train-*
  - split: test
    path: data/orc_Latn/test-*
  - split: dev
    path: data/orc_Latn/dev-*
- config_name: pbs_Latn
  data_files:
  - split: train
    path: data/pbs_Latn/train-*
  - split: test
    path: data/pbs_Latn/test-*
  - split: dev
    path: data/pbs_Latn/dev-*
- config_name: pbt_Arab
  data_files:
  - split: test
    path: data/pbt_Arab/test-*
  - split: dev
    path: data/pbt_Arab/dev-*
  - split: train
    path: data/pbt_Arab/train-*
- config_name: pbu_Arab
  data_files:
  - split: train
    path: data/pbu_Arab/train-*
  - split: test
    path: data/pbu_Arab/test-*
  - split: dev
    path: data/pbu_Arab/dev-*
- config_name: pex_Latn
  data_files:
  - split: train
    path: data/pex_Latn/train-*
  - split: test
    path: data/pex_Latn/test-*
  - split: dev
    path: data/pex_Latn/dev-*
- config_name: phr_Arab
  data_files:
  - split: train
    path: data/phr_Arab/train-*
  - split: dev
    path: data/phr_Arab/dev-*
  - split: test
    path: data/phr_Arab/test-*
- config_name: pip_Latn
  data_files:
  - split: train
    path: data/pip_Latn/train-*
  - split: dev
    path: data/pip_Latn/dev-*
  - split: test
    path: data/pip_Latn/test-*
- config_name: piy_Latn
  data_files:
  - split: test
    path: data/piy_Latn/test-*
  - split: train
    path: data/piy_Latn/train-*
  - split: dev
    path: data/piy_Latn/dev-*
- config_name: pko_Latn
  data_files:
  - split: test
    path: data/pko_Latn/test-*
  - split: train
    path: data/pko_Latn/train-*
  - split: dev
    path: data/pko_Latn/dev-*
- config_name: plt_Latn
  data_files:
  - split: train
    path: data/plt_Latn/train-*
- config_name: pmq_Latn
  data_files:
  - split: train
    path: data/pmq_Latn/train-*
  - split: dev
    path: data/pmq_Latn/dev-*
  - split: test
    path: data/pmq_Latn/test-*
- config_name: pms_Latn
  data_files:
  - split: train
    path: data/pms_Latn/train-*
- config_name: pmy_Latn
  data_files:
  - split: train
    path: data/pmy_Latn/train-*
  - split: dev
    path: data/pmy_Latn/dev-*
  - split: test
    path: data/pmy_Latn/test-*
- config_name: pnb_Arab
  data_files:
  - split: train
    path: data/pnb_Arab/train-*
  - split: test
    path: data/pnb_Arab/test-*
  - split: dev
    path: data/pnb_Arab/dev-*
- config_name: poc_Latn
  data_files:
  - split: test
    path: data/poc_Latn/test-*
  - split: dev
    path: data/poc_Latn/dev-*
  - split: train
    path: data/poc_Latn/train-*
- config_name: poe_Latn
  data_files:
  - split: train
    path: data/poe_Latn/train-*
  - split: dev
    path: data/poe_Latn/dev-*
  - split: test
    path: data/poe_Latn/test-*
- config_name: pow_Latn
  data_files:
  - split: train
    path: data/pow_Latn/train-*
  - split: test
    path: data/pow_Latn/test-*
  - split: dev
    path: data/pow_Latn/dev-*
- config_name: pst_Arab
  data_files:
  - split: test
    path: data/pst_Arab/test-*
  - split: train
    path: data/pst_Arab/train-*
  - split: dev
    path: data/pst_Arab/dev-*
- config_name: qug_Latn
  data_files:
  - split: dev
    path: data/qug_Latn/dev-*
  - split: train
    path: data/qug_Latn/train-*
  - split: test
    path: data/qug_Latn/test-*
- config_name: qum_Latn
  data_files:
  - split: train
    path: data/qum_Latn/train-*
  - split: dev
    path: data/qum_Latn/dev-*
  - split: test
    path: data/qum_Latn/test-*
- config_name: quv_Latn
  data_files:
  - split: train
    path: data/quv_Latn/train-*
  - split: test
    path: data/quv_Latn/test-*
  - split: dev
    path: data/quv_Latn/dev-*
- config_name: rag_Latn
  data_files:
  - split: test
    path: data/rag_Latn/test-*
  - split: train
    path: data/rag_Latn/train-*
  - split: dev
    path: data/rag_Latn/dev-*
- config_name: rob_Latn
  data_files:
  - split: train
    path: data/rob_Latn/train-*
  - split: dev
    path: data/rob_Latn/dev-*
  - split: test
    path: data/rob_Latn/test-*
- config_name: rof_Latn
  data_files:
  - split: test
    path: data/rof_Latn/test-*
  - split: train
    path: data/rof_Latn/train-*
  - split: dev
    path: data/rof_Latn/dev-*
- config_name: roo_Latn
  data_files:
  - split: train
    path: data/roo_Latn/train-*
  - split: test
    path: data/roo_Latn/test-*
  - split: dev
    path: data/roo_Latn/dev-*
- config_name: rth_Latn
  data_files:
  - split: train
    path: data/rth_Latn/train-*
  - split: dev
    path: data/rth_Latn/dev-*
  - split: test
    path: data/rth_Latn/test-*
- config_name: sau_Latn
  data_files:
  - split: dev
    path: data/sau_Latn/dev-*
  - split: test
    path: data/sau_Latn/test-*
  - split: train
    path: data/sau_Latn/train-*
- config_name: say_Latn
  data_files:
  - split: dev
    path: data/say_Latn/dev-*
  - split: test
    path: data/say_Latn/test-*
  - split: train
    path: data/say_Latn/train-*
- config_name: scn_Latn
  data_files:
  - split: train
    path: data/scn_Latn/train-*
  - split: test
    path: data/scn_Latn/test-*
  - split: dev
    path: data/scn_Latn/dev-*
- config_name: shu_Latn
  data_files:
  - split: train
    path: data/shu_Latn/train-*
- config_name: sin_Sinh
  data_files:
  - split: train
    path: data/sin_Sinh/train-*
  - split: test
    path: data/sin_Sinh/test-*
  - split: dev
    path: data/sin_Sinh/dev-*
- config_name: sip_Tibt
  data_files:
  - split: dev
    path: data/sip_Tibt/dev-*
  - split: train
    path: data/sip_Tibt/train-*
  - split: test
    path: data/sip_Tibt/test-*
- config_name: siw_Latn
  data_files:
  - split: train
    path: data/siw_Latn/train-*
  - split: test
    path: data/siw_Latn/test-*
  - split: dev
    path: data/siw_Latn/dev-*
- config_name: sjr_Latn
  data_files:
  - split: test
    path: data/sjr_Latn/test-*
  - split: train
    path: data/sjr_Latn/train-*
  - split: dev
    path: data/sjr_Latn/dev-*
- config_name: skg_Latn
  data_files:
  - split: train
    path: data/skg_Latn/train-*
- config_name: snc_Latn
  data_files:
  - split: test
    path: data/snc_Latn/test-*
  - split: train
    path: data/snc_Latn/train-*
  - split: dev
    path: data/snc_Latn/dev-*
- config_name: snk_Latn
  data_files:
  - split: dev
    path: data/snk_Latn/dev-*
  - split: train
    path: data/snk_Latn/train-*
  - split: test
    path: data/snk_Latn/test-*
- config_name: sol_Latn
  data_files:
  - split: dev
    path: data/sol_Latn/dev-*
  - split: train
    path: data/sol_Latn/train-*
  - split: test
    path: data/sol_Latn/test-*
- config_name: sps_Latn
  data_files:
  - split: train
    path: data/sps_Latn/train-*
  - split: dev
    path: data/sps_Latn/dev-*
  - split: test
    path: data/sps_Latn/test-*
- config_name: src_Latn
  data_files:
  - split: dev
    path: data/src_Latn/dev-*
  - split: train
    path: data/src_Latn/train-*
  - split: test
    path: data/src_Latn/test-*
- config_name: sro_Latn
  data_files:
  - split: train
    path: data/sro_Latn/train-*
  - split: dev
    path: data/sro_Latn/dev-*
  - split: test
    path: data/sro_Latn/test-*
- config_name: ste_Latn
  data_files:
  - split: train
    path: data/ste_Latn/train-*
  - split: test
    path: data/ste_Latn/test-*
  - split: dev
    path: data/ste_Latn/dev-*
- config_name: sua_Latn
  data_files:
  - split: train
    path: data/sua_Latn/train-*
  - split: dev
    path: data/sua_Latn/dev-*
  - split: test
    path: data/sua_Latn/test-*
- config_name: tan_Latn
  data_files:
  - split: train
    path: data/tan_Latn/train-*
  - split: dev
    path: data/tan_Latn/dev-*
  - split: test
    path: data/tan_Latn/test-*
- config_name: tbf_Latn
  data_files:
  - split: train
    path: data/tbf_Latn/train-*
  - split: dev
    path: data/tbf_Latn/dev-*
  - split: test
    path: data/tbf_Latn/test-*
- config_name: tcf_Latn
  data_files:
  - split: dev
    path: data/tcf_Latn/dev-*
  - split: train
    path: data/tcf_Latn/train-*
  - split: test
    path: data/tcf_Latn/test-*
- config_name: tcy_Mlym
  data_files:
  - split: test
    path: data/tcy_Mlym/test-*
  - split: train
    path: data/tcy_Mlym/train-*
  - split: dev
    path: data/tcy_Mlym/dev-*
- config_name: tdn_Latn
  data_files:
  - split: test
    path: data/tdn_Latn/test-*
  - split: dev
    path: data/tdn_Latn/dev-*
  - split: train
    path: data/tdn_Latn/train-*
- config_name: tdx_Latn
  data_files:
  - split: test
    path: data/tdx_Latn/test-*
  - split: train
    path: data/tdx_Latn/train-*
  - split: dev
    path: data/tdx_Latn/dev-*
- config_name: tgc_Latn
  data_files:
  - split: train
    path: data/tgc_Latn/train-*
  - split: test
    path: data/tgc_Latn/test-*
  - split: dev
    path: data/tgc_Latn/dev-*
- config_name: the_Deva
  data_files:
  - split: dev
    path: data/the_Deva/dev-*
  - split: train
    path: data/the_Deva/train-*
  - split: test
    path: data/the_Deva/test-*
- config_name: thq_Deva
  data_files:
  - split: dev
    path: data/thq_Deva/dev-*
  - split: train
    path: data/thq_Deva/train-*
  - split: test
    path: data/thq_Deva/test-*
- config_name: thr_Deva
  data_files:
  - split: train
    path: data/thr_Deva/train-*
  - split: dev
    path: data/thr_Deva/dev-*
  - split: test
    path: data/thr_Deva/test-*
- config_name: thv_Tfng
  data_files:
  - split: dev
    path: data/thv_Tfng/dev-*
  - split: train
    path: data/thv_Tfng/train-*
  - split: test
    path: data/thv_Tfng/test-*
- config_name: tio_Latn
  data_files:
  - split: train
    path: data/tio_Latn/train-*
  - split: dev
    path: data/tio_Latn/dev-*
  - split: test
    path: data/tio_Latn/test-*
- config_name: tkg_Latn
  data_files:
  - split: train
    path: data/tkg_Latn/train-*
  - split: test
    path: data/tkg_Latn/test-*
  - split: dev
    path: data/tkg_Latn/dev-*
- config_name: tkt_Deva
  data_files:
  - split: train
    path: data/tkt_Deva/train-*
  - split: dev
    path: data/tkt_Deva/dev-*
  - split: test
    path: data/tkt_Deva/test-*
- config_name: tlp_Latn
  data_files:
  - split: test
    path: data/tlp_Latn/test-*
  - split: train
    path: data/tlp_Latn/train-*
  - split: dev
    path: data/tlp_Latn/dev-*
- config_name: tpl_Latn
  data_files:
  - split: train
    path: data/tpl_Latn/train-*
  - split: dev
    path: data/tpl_Latn/dev-*
  - split: test
    path: data/tpl_Latn/test-*
- config_name: tpz_Latn
  data_files:
  - split: train
    path: data/tpz_Latn/train-*
  - split: dev
    path: data/tpz_Latn/dev-*
  - split: test
    path: data/tpz_Latn/test-*
- config_name: tqp_Latn
  data_files:
  - split: train
    path: data/tqp_Latn/train-*
  - split: test
    path: data/tqp_Latn/test-*
  - split: dev
    path: data/tqp_Latn/dev-*
- config_name: trp_Latn
  data_files:
  - split: train
    path: data/trp_Latn/train-*
  - split: dev
    path: data/trp_Latn/dev-*
  - split: test
    path: data/trp_Latn/test-*
- config_name: trq_Latn
  data_files:
  - split: train
    path: data/trq_Latn/train-*
  - split: dev
    path: data/trq_Latn/dev-*
  - split: test
    path: data/trq_Latn/test-*
- config_name: ttj_Latn
  data_files:
  - split: train
    path: data/ttj_Latn/train-*
  - split: test
    path: data/ttj_Latn/test-*
  - split: dev
    path: data/ttj_Latn/dev-*
- config_name: ttr_Latn
  data_files:
  - split: train
    path: data/ttr_Latn/train-*
  - split: test
    path: data/ttr_Latn/test-*
  - split: dev
    path: data/ttr_Latn/dev-*
- config_name: ttu_Latn
  data_files:
  - split: train
    path: data/ttu_Latn/train-*
  - split: dev
    path: data/ttu_Latn/dev-*
  - split: test
    path: data/ttu_Latn/test-*
- config_name: tul_Latn
  data_files:
  - split: train
    path: data/tul_Latn/train-*
  - split: test
    path: data/tul_Latn/test-*
  - split: dev
    path: data/tul_Latn/dev-*
- config_name: tuq_Latn
  data_files:
  - split: train
    path: data/tuq_Latn/train-*
  - split: dev
    path: data/tuq_Latn/dev-*
  - split: test
    path: data/tuq_Latn/test-*
- config_name: tuv_Latn
  data_files:
  - split: dev
    path: data/tuv_Latn/dev-*
  - split: train
    path: data/tuv_Latn/train-*
  - split: test
    path: data/tuv_Latn/test-*
- config_name: tuy_Latn
  data_files:
  - split: train
    path: data/tuy_Latn/train-*
  - split: test
    path: data/tuy_Latn/test-*
  - split: dev
    path: data/tuy_Latn/dev-*
- config_name: tvo_Latn
  data_files:
  - split: train
    path: data/tvo_Latn/train-*
  - split: test
    path: data/tvo_Latn/test-*
  - split: dev
    path: data/tvo_Latn/dev-*
- config_name: twu_Latn
  data_files:
  - split: train
    path: data/twu_Latn/train-*
  - split: dev
    path: data/twu_Latn/dev-*
  - split: test
    path: data/twu_Latn/test-*
- config_name: txs_Latn
  data_files:
  - split: train
    path: data/txs_Latn/train-*
  - split: test
    path: data/txs_Latn/test-*
  - split: dev
    path: data/txs_Latn/dev-*
- config_name: txy_Latn
  data_files:
  - split: dev
    path: data/txy_Latn/dev-*
  - split: test
    path: data/txy_Latn/test-*
  - split: train
    path: data/txy_Latn/train-*
- config_name: uki_Orya
  data_files:
  - split: test
    path: data/uki_Orya/test-*
  - split: dev
    path: data/uki_Orya/dev-*
  - split: train
    path: data/uki_Orya/train-*
- config_name: uzn_Latn
  data_files:
  - split: test
    path: data/uzn_Latn/test-*
  - split: train
    path: data/uzn_Latn/train-*
  - split: dev
    path: data/uzn_Latn/dev-*
- config_name: vai_Latn
  data_files:
  - split: test
    path: data/vai_Latn/test-*
  - split: train
    path: data/vai_Latn/train-*
  - split: dev
    path: data/vai_Latn/dev-*
- config_name: ver_Latn
  data_files:
  - split: train
    path: data/ver_Latn/train-*
  - split: test
    path: data/ver_Latn/test-*
  - split: dev
    path: data/ver_Latn/dev-*
- config_name: vmc_Latn
  data_files:
  - split: dev
    path: data/vmc_Latn/dev-*
  - split: train
    path: data/vmc_Latn/train-*
  - split: test
    path: data/vmc_Latn/test-*
- config_name: vmj_Latn
  data_files:
  - split: train
    path: data/vmj_Latn/train-*
  - split: dev
    path: data/vmj_Latn/dev-*
  - split: test
    path: data/vmj_Latn/test-*
- config_name: vmm_Latn
  data_files:
  - split: train
    path: data/vmm_Latn/train-*
  - split: dev
    path: data/vmm_Latn/dev-*
  - split: test
    path: data/vmm_Latn/test-*
- config_name: vmp_Latn
  data_files:
  - split: train
    path: data/vmp_Latn/train-*
  - split: test
    path: data/vmp_Latn/test-*
  - split: dev
    path: data/vmp_Latn/dev-*
- config_name: vmz_Latn
  data_files:
  - split: dev
    path: data/vmz_Latn/dev-*
  - split: train
    path: data/vmz_Latn/train-*
  - split: test
    path: data/vmz_Latn/test-*
- config_name: vro_Latn
  data_files:
  - split: train
    path: data/vro_Latn/train-*
  - split: dev
    path: data/vro_Latn/dev-*
  - split: test
    path: data/vro_Latn/test-*
- config_name: wci_Latn
  data_files:
  - split: dev
    path: data/wci_Latn/dev-*
  - split: train
    path: data/wci_Latn/train-*
  - split: test
    path: data/wci_Latn/test-*
- config_name: weo_Latn
  data_files:
  - split: test
    path: data/weo_Latn/test-*
  - split: dev
    path: data/weo_Latn/dev-*
  - split: train
    path: data/weo_Latn/train-*
- config_name: wja_Latn
  data_files:
  - split: train
    path: data/wja_Latn/train-*
  - split: dev
    path: data/wja_Latn/dev-*
  - split: test
    path: data/wja_Latn/test-*
- config_name: wji_Latn
  data_files:
  - split: train
    path: data/wji_Latn/train-*
  - split: dev
    path: data/wji_Latn/dev-*
  - split: test
    path: data/wji_Latn/test-*
- config_name: wof_Latn
  data_files:
  - split: train
    path: data/wof_Latn/train-*
  - split: dev
    path: data/wof_Latn/dev-*
  - split: test
    path: data/wof_Latn/test-*
- config_name: xmv_Latn
  data_files:
  - split: test
    path: data/xmv_Latn/test-*
  - split: train
    path: data/xmv_Latn/train-*
  - split: dev
    path: data/xmv_Latn/dev-*
- config_name: xmw_Latn
  data_files:
  - split: train
    path: data/xmw_Latn/train-*
- config_name: xpe_Latn
  data_files:
  - split: train
    path: data/xpe_Latn/train-*
  - split: test
    path: data/xpe_Latn/test-*
  - split: dev
    path: data/xpe_Latn/dev-*
- config_name: xti_Latn
  data_files:
  - split: train
    path: data/xti_Latn/train-*
  - split: test
    path: data/xti_Latn/test-*
  - split: dev
    path: data/xti_Latn/dev-*
- config_name: xtu_Latn
  data_files:
  - split: train
    path: data/xtu_Latn/train-*
  - split: test
    path: data/xtu_Latn/test-*
  - split: dev
    path: data/xtu_Latn/dev-*
- config_name: yay_Latn
  data_files:
  - split: train
    path: data/yay_Latn/train-*
  - split: dev
    path: data/yay_Latn/dev-*
  - split: test
    path: data/yay_Latn/test-*
- config_name: ydd_Hebr
  data_files:
  - split: train
    path: data/ydd_Hebr/train-*
  - split: dev
    path: data/ydd_Hebr/dev-*
  - split: test
    path: data/ydd_Hebr/test-*
- config_name: yer_Latn
  data_files:
  - split: dev
    path: data/yer_Latn/dev-*
  - split: train
    path: data/yer_Latn/train-*
  - split: test
    path: data/yer_Latn/test-*
- config_name: yes_Latn
  data_files:
  - split: train
    path: data/yes_Latn/train-*
  - split: dev
    path: data/yes_Latn/dev-*
  - split: test
    path: data/yes_Latn/test-*
- config_name: zga_Latn
  data_files:
  - split: train
    path: data/zga_Latn/train-*
- config_name: zoh_Latn
  data_files:
  - split: train
    path: data/zoh_Latn/train-*
  - split: test
    path: data/zoh_Latn/test-*
  - split: dev
    path: data/zoh_Latn/dev-*
- config_name: zor_Latn
  data_files:
  - split: train
    path: data/zor_Latn/train-*
  - split: test
    path: data/zor_Latn/test-*
  - split: dev
    path: data/zor_Latn/dev-*
- config_name: zpv_Latn
  data_files:
  - split: test
    path: data/zpv_Latn/test-*
  - split: train
    path: data/zpv_Latn/train-*
  - split: dev
    path: data/zpv_Latn/dev-*
- config_name: zpy_Latn
  data_files:
  - split: dev
    path: data/zpy_Latn/dev-*
  - split: train
    path: data/zpy_Latn/train-*
  - split: test
    path: data/zpy_Latn/test-*
- config_name: ztg_Latn
  data_files:
  - split: train
    path: data/ztg_Latn/train-*
  - split: dev
    path: data/ztg_Latn/dev-*
  - split: test
    path: data/ztg_Latn/test-*
- config_name: ztn_Latn
  data_files:
  - split: train
    path: data/ztn_Latn/train-*
  - split: dev
    path: data/ztn_Latn/dev-*
  - split: test
    path: data/ztn_Latn/test-*
- config_name: ztp_Latn
  data_files:
  - split: train
    path: data/ztp_Latn/train-*
  - split: test
    path: data/ztp_Latn/test-*
  - split: dev
    path: data/ztp_Latn/dev-*
- config_name: zts_Latn
  data_files:
  - split: train
    path: data/zts_Latn/train-*
  - split: dev
    path: data/zts_Latn/dev-*
  - split: test
    path: data/zts_Latn/test-*
- config_name: ztu_Latn
  data_files:
  - split: dev
    path: data/ztu_Latn/dev-*
  - split: test
    path: data/ztu_Latn/test-*
  - split: train
    path: data/ztu_Latn/train-*
task_categories:
- automatic-speech-recognition
- audio-classification
pretty_name: Omnilingual ASR Corpus
size_categories:
- 100K<n<1M
---

# Meta Omnilingual ASR Corpus

The Omnilingual ASR Corpus is a collection of spontaneous speech recordings and their transcriptions for 348 under-served languages. The corpus was collected as part of Meta FAIR’s Omnilingual ASR project ([blog](https://ai.meta.com/blog/omnilingual-asr-advancing-automatic-speech-recognition/), [model](https://github.com/facebookresearch/omnilingual-asr), [paper](https://ai.meta.com/research/publications/omnilingual-asr-open-source-multilingual-speech-recognition-for-1600-languages/)) for the purposes of training automatic speech recognition (ASR) and spoken language identification models.

## Data schema

```json
{
    `language`: "lij_Latn",
    `iso_639_3`: "lij",
    `iso_15924`: "Latn",
    `glottocode`: "geno1240",
    `prompt_id`: "C086",
    `prompt`: "What was the last thing you ate? Can you describe how it is made?",
    `speaker_id`: "spk02",
    `segment_id`: "s01",
    `audio`: "<Audio data in FLAC format>",
    `raw_text`: "Me son tòsto fæto un panetto co-o formaggio, ma quello a-a catalaña, saiva à dî con o pan un pittin brustolio e pöi a tomata sciaccâ in çimma, tanto euio e un pittin de sâ, e dapeu se ghe mette o companægo, into mæ caxo o formaggio.",
}
```

## Language codes

Language codes in the `language` column follow the format `{lang}_{script}`, where `{lang}` is an ISO 639-3 three-letter language code, and `{script}` is an ISO 15924 four-letter script code. To allow for greater granularity when warranted, we provide the additional `glottocode` column, containing [Glottolog](http://glottolog.org/) languoid codes.

## Special tags

The following special tags were used in transcriptions (`raw_text` field) to mark laughter, fillers and other types of non-verbal content:

| Tag                | Purpose     |
|--------------------|-------------|
| `<laugh>`          | The sound of laughter. |
| `<hesitation>`     | A hesitation sound, often used by speakers while thinking of the next thing to say. In English, some common hesitation sounds are “err”, “um”, “huh”, etc. |
| `<unintelligible>` | A word or sequence of words that cannot be understood. |
| `<noise>`          | Any other type of noise, such as the speaker coughing or clearing their throat, a car honking, the sound of something hitting the microphone, a phone buzzing, etc. |

## Disfluencies

Spontaneous speech naturally contains false starts, where only a fragment of a full word is produced. False starts were transcribed as they appeared in the recording and a hyphen was attached at the end of the word fragment (-), e.g.:

> His name is Jo- Jona- Jonathan.

Repeated words were also faithfully transcribed, e.g.:

> And then I went to the the the bed- the bedroom

## License

This corpus is released under CC-BY-4.0.

## Citation

If you make use of this dataset in your work, please cite:

```bibtex
@misc{omnilingualasr2025,
    title={{Omnilingual ASR}: Open-Source Multilingual Speech Recognition for 1600+ Languages},
    author={{Omnilingual ASR Team} and Keren, Gil and Kozhevnikov, Artyom and Meng, Yen and Ropers, Christophe and Setzler, Matthew and Wang, Skyler and Adebara, Ife and Auli, Michael and Balioglu, Can and Chan, Kevin and Cheng, Chierh and Chuang, Joe and Droof, Caley and Duppenthaler, Mark and Duquenne, Paul-Ambroise and Erben, Alexander and Gao, Cynthia and Mejia Gonzalez, Gabriel and Lyu, Kehan and Miglani, Sagar and Pratap, Vineel and Sadagopan, Kaushik Ram and Saleem, Safiyyah and Turkatenko, Arina and Ventayol-Boada, Albert and Yong, Zheng-Xin and Chung, Yu-An and Maillard, Jean and Moritz, Rashel and Mourachko, Alexandre and Williamson, Mary and Yates, Shireen},
    year={2025},
    eprint={2511.09690},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2511.09690}, 
}
```