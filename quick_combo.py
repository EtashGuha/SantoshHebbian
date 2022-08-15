import pickle

with open("/nethome/eguha3/SantoshHebbian/new_code_smooth_n_2p_con.pkl", "rb") as f:
	just_con = pickle.load(f)

with open("/nethome/eguha3/SantoshHebbian/new_code_smooth_n_2p.pkl", 'rb') as f:
	both = pickle.load(f)


both.update(just_con)

with open("/nethome/eguha3/SantoshHebbian/new_code_smooth_n_2p_all.pkl", "wb") as otherf:
	pickle.dump(both, otherf)