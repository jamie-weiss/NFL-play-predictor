import pandas
import feature_engineering as fe

df = fe.read_csv('final.csv')

i = 0
for play in df['PlayType']:
	if play not in ["RUSH", "PASS"]:
		print(i, play)
	i = i + 1
