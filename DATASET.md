# Dataset provenance and usage

HookRec uses the **HetRec 2011 Last.fm 2K dataset**, hosted by GroupLens.

- Official page: https://grouplens.org/datasets/hetrec-2011/
- Archive: https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip

The archive contains user-artist listening counts, artist metadata, user-applied artist tags and tag names. The dataset README states that the data is made available for **non-commercial use**. Commercial use requires contacting the relevant rights holder. The MIT license for HookRec source code does not apply to the dataset.

Files used: `user_artists.dat`, `artists.dat`, `user_taggedartists.dat`, `tags.dat`. Dataset files are downloaded at setup time and are not committed to this repository.

Please cite Last.fm (https://www.last.fm/) and Cantador, I., Brusilovsky, P., & Kuflik, T. (2011), *2nd Workshop on Information Heterogeneity and Fusion in Recommender Systems (HetRec 2011)*.
