# Data-Related-Projects
##### 整个项目包含了跟数据相关的project，由于github对文件大小的限制，数据格式无法上传
## 数据文件格式
- genome-tags.csv <img src="https://github.com/ryang1995/Data-Related-Projects/tree/master/pictures/genome-tags.png" alt="genome-tags data format" title="snapshot1">
- imdb-actor-info.csv <img src="https://github.com/ryang1995/Data-Related-Projects/tree/master/pictures/imdb-actor-info.png" alt="imdb-actor-info data format" title="snapshot2">
- mlmovies.csv <img src="https://github.com/ryang1995/Data-Related-Projects/tree/master/pictures/mlmovies.png" alt="mlmovies data format" title="snapshot3">
- mlratings.csv <img src="https://github.com/ryang1995/Data-Related-Projects/tree/master/pictures/mlratings.png" alt="mlratings data format" title="snapshot4">
- mltags.csv <img src="https://github.com/ryang1995/Data-Related-Projects/tree/master/pictures/mltags.png" alt="mltags data format" title="snapshot5">
- mlusers.csv <img src="https://github.com/ryang1995/Data-Related-Projects/tree/master/pictures/mlusers.png" alt="mlusers data format" title="snapshot6">
- movie-actor.csv <img src="https://github.com/ryang1995/Data-Related-Projects/tree/master/pictures/movie-actor.png" alt="movie-actor data format" title="snapshot7">
## 文件内容
- phase1: 对所给的数据集，根据不同的属性，建立带权值的TF/ TF-IDF模型，建立TF-IDF-DIFF, P-DIFF1, P-DIFF2模型。
- phase2: 根据所给的数据集，实现重启随机游走算法（PWP）来做相似度推荐，比如给定目的电影，推荐相似度最高的10部电影。
- phase3: 对所给的数据集，给定一个属性，实现了用PCA, SVD, LDA方法获得潜在语义／属性。运用PCA, SVD, 张量分解等方法对数据集进行降维处理，进而做相似度推荐。
