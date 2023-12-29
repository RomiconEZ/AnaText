# AnaText
## Description/Описание:
### RU:
Программа «AnaText» предназначена для
* решения задачи разведочного анализа текстов с применением модели [Supporting Clustering with Contrastive Learning](https://github.com/amazon-science/sccl) для кластеризации 
* подбора гиперпараметров через оптимизацию функционала 
* выделения ключевых слов для получившихся кластеров 
* взаимодействия с полученной кластеризацией
* построения модели классификации для полученного разбиения  

Предоставляет доступный пользовательский интерфейс, способствующий упрощению процесса анализа данных и интерпретации результатов.

К преимуществам данной программы (по отношению к известным) относятся:

* Низкий порог входа для начала использования;
* Обработка любой текстовой информации; 
* Интерактивная работа с данными: пользователь может редактировать кластерную структуру, которая была получена в ходе обработки; 
* Дообучение модели кластеризации по требованию пользователя на основании получившегося разбиения текстов;
* Отсутствие необходимости ручного подбора параметров для функций кластеризации, подбора ключевых слов и аппроксимации числа кластеров;
* Инкапсуляция всех этапов загрузки, обработки и постобработки текстовой информации в виде единого интерфейса;
* Поддержка Metal Performance Shaders backend;


### ENG:
The AnaText program is designed for
* solving the problem of exploratory text analysis using the [Supporting Clustering with Contrastive Learning model](https://github.com/amazon-science/sccl) for clustering
* selection of hyperparameters through optimization of functionality
* highlighting keywords for the resulting clusters
* the ability to interact with the resulting clustering
* building a classification model for the resulting partition

Provides accessible user interface that simplifies the process of data analysis and interpretation of results.

The advantages of this program (in relation to the known ones) include:

* Low entry threshold to start using;
* Processing of any text information;
* Interactive work with data: the user can edit the cluster structure that was obtained during processing;
* Additional training of the clustering model at the user's request based on the resulting text splitting;
* No need for manual selection of parameters for clustering functions, keyword selection and approximation of the number of clusters;
* Encapsulation of all stages of loading, processing and postprocessing of textual information in a single interface;
* Support for Metal Performance Shaders backend;

## Documentation/Документация:
https://romiconez.github.io/AnaText/

## How to use/Как использовать:

(Data: https://github.com/RomiconEZ/AnaText/blob/main/tests/20_newsgroup_text_only_50.csv)

    pip install AnaText

    =======================================

    from pathlib import Path
    import eta

    def test_split_merge_clusters():
        current_path = Path(__file__).parent
        filename = current_path / "20_newsgroup_text_only_50.csv"
    
        output_dict = eta.cluster_documents_with_keywords(filename, verbose=True)
    
        df = output_dict["df"]
        top_word_dict = output_dict["top_word_dict"]
        data = output_dict["data"]
        cluster_centers_1 = output_dict["cluster_centers"]
        radiuses = output_dict["radiuses"]
        cluster_model_1 = output_dict["cluster_model"]
        cluster_centers_2d_1 = output_dict["cluster_centers_2d"]
        reduce_model_1 = output_dict["reduce_model"]
        embeddings_1 = output_dict["embeddings"]
        tokenizer = output_dict["tokenizer"]
        model = output_dict["model"]
    
        cl_list = [0, 1]
        cluster_num = 0
        divisor = 2
    
        output_dict_split = eta.split_cluster(cluster_num, divisor, data, reduce_model_1, embeddings_1.to_list())
        data = output_dict_split["data"]
        cluster_centers_2d = output_dict_split["cluster_centers_2d"]
        radiuses = output_dict_split["radiuses"]
    
        output_dict_union = eta.union_clusters(cl_list, data, reduce_model_1, embeddings_1.to_list())
        data = output_dict_union["data"]
        cluster_centers_2d = output_dict_union["cluster_centers_2d"]
        radiuses = output_dict_union["radiuses"]
    
        return
    
    if __name__ == '__main__':
        test_split_merge_clusters()