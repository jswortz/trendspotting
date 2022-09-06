import kfp
from typing import Dict, List, Optional, Sequence, Tuple, Union
from kfp.v2.dsl import Artifact
from kfp.v2.dsl import Input, Model
from kfp.v2.components.types.type_utils import artifact_types
from typing import Any, Callable, Dict, NamedTuple, Optional


@kfp.v2.dsl.component(
  base_image='python:3.9',
  packages_to_install=['google-cloud-bigquery==2.18.0'],
)
def create_prediction_dataset_term_level(
      target_table: str,
      source_table_uri: str,
      train_st: str,
      train_end: str,
      valid_st: str,
      valid_end: str,
      subcat_id: int,
      override: str = 'False',
      project_id: str = 'cpg-cdp'
    ) -> NamedTuple('Outputs', [('training_data_table_uri', str)]):
    
    from google.cloud import bigquery
 
    override = bool(override)
    bq_client = bigquery.Client(project=project_id)
    (
    bq_client.query(
      f"""CREATE TEMPORARY FUNCTION arr_to_input_20(arr ARRAY<FLOAT64>)
        RETURNS 
        STRUCT<p1 FLOAT64, p2 FLOAT64, p3 FLOAT64, p4 FLOAT64,
               p5 FLOAT64, p6 FLOAT64, p7 FLOAT64, p8 FLOAT64, 
               p9 FLOAT64, p10 FLOAT64, p11 FLOAT64, p12 FLOAT64, 
               p13 FLOAT64, p14 FLOAT64, p15 FLOAT64, p16 FLOAT64,
               p17 FLOAT64, p18 FLOAT64, p19 FLOAT64, p20 FLOAT64>
        AS (
        STRUCT(
            arr[OFFSET(0)]
            , arr[OFFSET(1)]
            , arr[OFFSET(2)]
            , arr[OFFSET(3)]
            , arr[OFFSET(4)]
            , arr[OFFSET(5)]
            , arr[OFFSET(6)]
            , arr[OFFSET(7)]
            , arr[OFFSET(8)]
            , arr[OFFSET(9)]
            , arr[OFFSET(10)]
            , arr[OFFSET(11)]
            , arr[OFFSET(12)]
            , arr[OFFSET(13)]
            , arr[OFFSET(14)]
            , arr[OFFSET(15)]
            , arr[OFFSET(16)]
            , arr[OFFSET(17)]
            , arr[OFFSET(18)]
            , arr[OFFSET(19)]    
        ));


        CREATE OR REPLACE TABLE `{target_table}` as (
            SELECT * except(output_0), case when date between "{train_st}" and "{train_end}" then 'TRAIN'
                  when date between "{valid_st}" and "{valid_end}" then 'VALIDATE'
                 else 'TEST' end as split_col,
            arr_to_input_20(output_0) as embed
        FROM ML.PREDICT(MODEL trendspotting.swivel_text_embed,
        (
          SELECT date, geo_id, term AS sentences, category_rank, concat( term, geo_id) as series_id
          FROM `{source_table_uri}` where category_id = {subcat_id} and date > "{train_st}"
        ))      
        )
          """
    )
    .result()
    )

    return (
    f'{target_table}',
    )


@kfp.v2.dsl.component(
  base_image='python:3.9',
  packages_to_install=['google-cloud-bigquery==2.18.0'],
)
def prep_forecast_term_level(
    source_table: str,
    target_table: str,
    override: str = 'False',
    project_id: str = 'cpg-cdp'
    ) -> NamedTuple('Outputs', [('term_train_table', str)]):
    
    from google.cloud import bigquery

    bq_client = bigquery.Client(project=project_id)
    (
    bq_client.query(
      f"""
            CREATE OR REPLACE TABLE `{target_table}` as (
        SELECT * except(embed), 
        embed.p1 as emb1, 
        embed.p2 as emb2,
        embed.p3 as emb3,
        embed.p4 as emb4,
        embed.p5 as emb5,
        embed.p6 as emb6,
        embed.p7 as emb7,
        embed.p8 as emb8,
        embed.p9 as emb9,
        embed.p10 as emb10,
        embed.p11 as emb11,
        embed.p12 as emb12,
        embed.p13 as emb13,
        embed.p14 as emb14,
        embed.p15 as emb15,
        embed.p16 as emb16,
        embed.p17 as emb17,
        embed.p18 as emb18,
        embed.p19 as emb19,
        embed.p20 as emb20

        FROM `{source_table}` )
          """
    )
    .result()
    )

    return (
    f'bq://{target_table}',
    )

@kfp.v2.dsl.component(
  base_image='python:3.9',
  packages_to_install=['google-cloud-bigquery==2.18.0'],
)
def create_top_mover_table(
    source_table: str,
    target_table: str,
    predict_on_dt: str, #uses the last validation date,
    six_month_dt: str,
    trained_model: Input[Artifact],
    top_n_results: int,
    override: str = 'False',
    project_id: str = 'cpg-cdp'
    ) -> NamedTuple('Outputs', [('term_train_table', str)]):
    
    from google.cloud import bigquery
    
    source_table_no_bq = source_table.strip('bq://')

    bq_client = bigquery.Client(project=project_id)
    (
    bq_client.query(
      f"""
            CREATE OR REPLACE TABLE {target_table} as (
    select * from
      (with six_mo_val as (select *, predicted_category_rank.value as six_mo_forecast from `{source_table_no_bq}` 
        where predicted_on_date = '{predict_on_dt}' and date = '{six_month_dt}'),
         geo_id as (select distinct geo_id, geo_name from `cpg-cdp.trendspotting.futurama_weekly`)
    SELECT a.date, 
       geo_id.geo_name, 
       a.sentences, 
       cast(a.category_rank as int64) as current_rank, 
       cast(a.category_rank as int64) - b.six_mo_forecast as six_delta_rank,
       cast(b.category_rank as int64) as six_mo_rank, 
       six_mo_forecast
      FROM `{source_table_no_bq}` a INNER JOIN 
       six_mo_val b on a.series_id = b.series_id 
       inner join 
       geo_id on cast(a.geo_id as int64) = geo_id.geo_id
      WHERE a.date = '{predict_on_dt}'
      ) where current_rank > 500 and six_mo_forecast < 1000 order by six_delta_rank desc limit {top_n_results} 
)
          """
    )
    .result()
    )

    return (
    f'{target_table}',
    )

@kfp.v2.dsl.component(
  base_image='python:3.9',
  packages_to_install=['google-cloud-bigquery==2.18.0'],
)
def nlp_featurize_and_cluster(
    source_table: str,
    target_table: str,
    train_st: str,
    train_end: str,
    subcat_id: int,
    model_name: str,
    override: str = 'False',
    project_id: str = 'cpg-cdp'
    ) -> NamedTuple('Outputs', [('term_cluster_table', str)]):
    
    from google.cloud import bigquery
    
    bq_client = bigquery.Client(project=project_id)
    (
    bq_client.query(
      f"""
            CREATE TEMPORARY FUNCTION arr_to_input_20(arr ARRAY<FLOAT64>)
            RETURNS 
            STRUCT<p1 FLOAT64, p2 FLOAT64, p3 FLOAT64, p4 FLOAT64,
                   p5 FLOAT64, p6 FLOAT64, p7 FLOAT64, p8 FLOAT64, 
                   p9 FLOAT64, p10 FLOAT64, p11 FLOAT64, p12 FLOAT64, 
                   p13 FLOAT64, p14 FLOAT64, p15 FLOAT64, p16 FLOAT64,
                   p17 FLOAT64, p18 FLOAT64, p19 FLOAT64, p20 FLOAT64>
            AS (
            STRUCT(
                arr[OFFSET(0)]
                , arr[OFFSET(1)]
                , arr[OFFSET(2)]
                , arr[OFFSET(3)]
                , arr[OFFSET(4)]
                , arr[OFFSET(5)]
                , arr[OFFSET(6)]
                , arr[OFFSET(7)]
                , arr[OFFSET(8)]
                , arr[OFFSET(9)]
                , arr[OFFSET(10)]
                , arr[OFFSET(11)]
                , arr[OFFSET(12)]
                , arr[OFFSET(13)]
                , arr[OFFSET(14)]
                , arr[OFFSET(15)]
                , arr[OFFSET(16)]
                , arr[OFFSET(17)]
                , arr[OFFSET(18)]
                , arr[OFFSET(19)]    
            ));

            CREATE OR REPLACE TABLE `{target_table}` as ( #
                select * 
                from ML.PREDICT(MODEL `{model_name}`, (
                    select *, arr_to_input_20(output_0) AS comments_embed from 
                        ML.PREDICT(MODEL trendspotting.swivel_text_embed,(
                      SELECT date, geo_name, term AS sentences, volume
                      FROM `{source_table}`
                      WHERE date >= '{train_st}'
                      and category_id = {subcat_id}
                    ))
                    )
                )
            )
          """
    )
    .result()
    )

    return (
    f'{target_table}',
    )

@kfp.v2.dsl.component(
  base_image='python:3.9',
  packages_to_install=['google-cloud-bigquery==2.18.0'],
)
def aggregate_clusters(
    source_table: str,
    target_table: str,
    train_st: str,
    train_end: str,
    valid_st: str,
    valid_end: str,
    model_name: str,
    override: str = 'False',
    project_id: str = 'cpg-cdp'
    ) -> NamedTuple('Outputs', [('term_cluster_agg_table', str)]):
    
    from google.cloud import bigquery
    
    source_table_no_bq = source_table.strip('bq://')
    
    target_bq_table = 'bq://' + target_table

    bq_client = bigquery.Client(project=project_id)
    (
    bq_client.query(
      f"""
            CREATE OR REPLACE TABLE `{target_table}` as (
            with centroids as (select * from 
            (SELECT
            centroid_id, feature, numerical_value
            FROM
              ML.CENTROIDS(MODEL `{model_name}`)
            )
            PIVOT(avg(numerical_value) for feature in ('comments_embed_p1',
            'comments_embed_p2',
            'comments_embed_p3',
            'comments_embed_p4',
            'comments_embed_p5',
            'comments_embed_p6',
            'comments_embed_p7',
            'comments_embed_p8',
            'comments_embed_p9',
            'comments_embed_p10',
            'comments_embed_p11',
            'comments_embed_p12',
            'comments_embed_p13',
            'comments_embed_p14',
            'comments_embed_p15',
            'comments_embed_p16',
            'comments_embed_p17',
            'comments_embed_p18',
            'comments_embed_p19',
            'comments_embed_p20'))
                              )
            select volume, date, b.*,
            case when date between '{train_st}' and  '{train_end}' then 'TRAIN'
                      when date between '{valid_st}' and '{valid_end}' then 'VALIDATE'
                     else 'TEST' end as split_col
            from (
                select sum(volume) as volume, date, centroid_id 
                from {source_table} group by date, centroid_id
            ) a
            inner join centroids b on a.centroid_id = b.centroid_id
            )
          """
    )
    .result()
    )

    return (
    f'{target_bq_table}',
    )

COLUMN_TRANSFORMS_CLUSTER = [
  {
    "numeric": {
      "columnName": "volume"
    }
  },
  {
    "timestamp": {
      "columnName": "date"
    }
  },
  {
    "numeric": {
      "columnName": "comments_embed_p1"
    }
  },
  {
    "numeric": {
      "columnName": "comments_embed_p2"
    }
  },
  {
    "numeric": {
      "columnName": "comments_embed_p3"
    }
  },
  {
    "numeric": {
      "columnName": "comments_embed_p4"
    }
  },
  {
    "numeric": {
      "columnName": "comments_embed_p5"
    }
  },
  {
    "numeric": {
      "columnName": "comments_embed_p6"
    }
  },
  {
    "numeric": {
      "columnName": "comments_embed_p7"
    }
  },
  {
    "numeric": {
      "columnName": "comments_embed_p8"
    }
  },
  {
    "numeric": {
      "columnName": "comments_embed_p9"
    }
  },
  {
    "numeric": {
      "columnName": "comments_embed_p10"
    }
  },
  {
    "numeric": {
      "columnName": "comments_embed_p11"
    }
  },
  {
    "numeric": {
      "columnName": "comments_embed_p12"
    }
  },
  {
    "numeric": {
      "columnName": "comments_embed_p13"
    }
  },
  {
    "numeric": {
      "columnName": "comments_embed_p14"
    }
  },
  {
    "numeric": {
      "columnName": "comments_embed_p15"
    }
  },
  {
    "numeric": {
      "columnName": "comments_embed_p16"
    }
  },
  {
    "numeric": {
      "columnName": "comments_embed_p17"
    }
  },
  {
    "numeric": {
      "columnName": "comments_embed_p18"
    }
  },
  {
    "numeric": {
      "columnName": "comments_embed_p19"
    }
  },
  {
    "numeric": {
      "columnName": "comments_embed_p20"
    }
  }
]

COLUMN_TRANSFORMATIONS = [
  {
    "timestamp": {
      "columnName": "date"
    }
  },
  {
    "categorical": {
      "columnName": "geo_id"
    }
  },
  {
    "text": {
      "columnName": "sentences"
    }
  },
  {
    "numeric": {
      "columnName": "category_rank"
    }
  },
  {
    "numeric": {
      "columnName": "emb1"
    }
  },
  {
    "numeric": {
      "columnName": "emb2"
    }
  },
  {
    "numeric": {
      "columnName": "emb3"
    }
  },
  {
    "numeric": {
      "columnName": "emb4"
    }
  },
  {
    "numeric": {
      "columnName": "emb5"
    }
  },
  {
    "numeric": {
      "columnName": "emb6"
    }
  },
  {
    "numeric": {
      "columnName": "emb7"
    }
  },
  {
    "numeric": {
      "columnName": "emb8"
    }
  },
  {
    "numeric": {
      "columnName": "emb9"
    }
  },
  {
    "numeric": {
      "columnName": "emb10"
    }
  },
  {
    "numeric": {
      "columnName": "emb11"
    }
  },
  {
    "numeric": {
      "columnName": "emb12"
    }
  },
  {
    "numeric": {
      "columnName": "emb13"
    }
  },
  {
    "numeric": {
      "columnName": "emb14"
    }
  },
  {
    "numeric": {
      "columnName": "emb15"
    }
  },
  {
    "numeric": {
      "columnName": "emb16"
    }
  },
  {
    "numeric": {
      "columnName": "emb17"
    }
  },
  {
    "numeric": {
      "columnName": "emb18"
    }
  },
  {
    "numeric": {
      "columnName": "emb19"
    }
  },
  {
    "numeric": {
      "columnName": "emb20"
    }
  }
]


@kfp.v2.dsl.component(
  base_image='python:3.9',
  packages_to_install=['google-cloud-bigquery==2.18.0','pandas','pyarrow'],
)

def auto_cluster(
    cluster_min: int,
    cluster_max: int,
    labels: list,
    source_table: str = "trendspotting.cat_clus_train_png_hair_22",
    target_table: str = "trendspotting.full_cat_clus_png_hair_22",
    project_id: str = 'cpg-cdp'
    ) -> NamedTuple('Outputs', [('target_table', str)]):
    
    from google.cloud import bigquery

    
    bq_client = bigquery.Client(project_id)

    ## use this function to get the name of the topic in the clustering
    def only_upper(s: str):
        upper_chars = ""
        for char in s:
            if char.isupper():
                upper_chars += char
        return upper_chars
    
        ## we use this to find where the DB index flattens for n_clusters and use that for optimal number of clusters per topic

    def loop_n_clus_and_get_db_index(cluster_min: int, cluster_max: int, label: str):

        label_upper = only_upper(label) #get only the upper case letters to denote the model name
        return_data = {label: []}
        for n_clusters in range(cluster_min, cluster_max+1):
            print(f"Training for {n_clusters} clusters")
            # return_data[label].append({'model_name': f'trendspotting.cat_clus_{label_upper}_{n_clusters}_png_hair_22'})
            kmeans_sql = f"""
            CREATE OR REPLACE MODEL trendspotting.cat_clus_{label_upper}_{n_clusters}_png_hair_22
            OPTIONS(model_type='kmeans', num_clusters={n_clusters}, standardize_features = true) AS
            select * EXCEPT(predicted_label, sentences) from {source_table}
            WHERE predicted_label = '{label}'
            """
            bq_client.query(kmeans_sql).result()
            #next, get the DB index to assess the cluster quality
            sql = f"""
            SELECT
              *
            FROM
              ML.EVALUATE (MODEL trendspotting.cat_clus_{label_upper}_{n_clusters}_png_hair_22)
              """
            data = bq_client.query(sql).to_dataframe()
            print(f"DB Index: {data.davies_bouldin_index[0]}")
            return_data[label].append({f'trendspotting.cat_clus_{label_upper}_{n_clusters}_png_hair_22': data.davies_bouldin_index[0]})

            time.sleep(60)

        return(return_data)
    data_dict = {}
    
    #loop over labels
    for label in labels:
        print(f"Tranining for label: {label}")
        cluster_data = loop_n_clus_and_get_db_index(cluster_min, cluster_max, label)
        data_dict.update(cluster_data) #update with the results
        time.sleep(60)
        
    # find the min DB score cluster for each topic, delete the other models and then score based on topic id

    optimal_models_by_label = {}
    for label in labels:
        prior_db=999 # set this high
        for c in data_dict[label]:
            if list(c.values())[0] < prior_db:
                prior_db = list(c.values())[0]
                optimal_model = list(c.keys())[0]         
        print(optimal_model)
        optimal_models_by_label.update({label: optimal_model})
    print(f"Optimal models found: {optimal_models_by_label}")
    
    # save optimal model dictionary to gcs
    from google.cloud import storage

    import pickle

    with open('./optimal_models.dict', 'wb') as file:
        pickle.dump(optimal_models_by_label, file)

    bucket_name = 'trendspotting-pipeline'

    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob('optimal_models.dict')

    blob.upload_from_filename('optimal_models.dict')
        
    print("Deleting sub optimal models")
    
    #delete the sub-optimal models
    def delete_model_sql(model_name):
        return f"DROP MODEL IF EXISTS {model_name}"

    for label in labels.predicted_label:
        optimal_model_for_label = optimal_models_by_label[label]
        for c in data_dict[label]:
            if list(c.keys())[0] != optimal_model_for_label:
                sub_optimal_model = list(c.keys())[0]
                bq_client.query(delete_model_sql(sub_optimal_model)).result() #clean up the models
                time.sleep(5)
                
   #last, score using a union query for each label

    def score_cluster(label, model_name):
        predict_sql = f"""
                SELECT
                  *
                FROM
                  ML.PREDICT (MODEL {model_name},
                  (SELECT * EXCEPT(predicted_label, sentences), 
                  sentences as terms, 
                  predicted_label as category 
                  from trendspotting.cat_clus_train_png_hair_22 
                  where predicted_label = '{label}'))
                  """
        return(predict_sql)

    predict_sql = ""
    for i, label in enumerate(labels.predicted_label):
        predict_sql += score_cluster(label, optimal_models_by_label[label])
        if len(labels.predicted_label)-1 == i:
            break
        else:
            predict_sql += """
            UNION ALL
            """

    def score_table(predict_sql, target_table="trendspotting.full_cat_clus_png_hair_22"):
        return(f"CREATE OR REPLACE TABLE {target_table} AS ({predict_sql})")

    segment_score_sql = score_table(predict_sql)

    bq_client.query(segment_score_sql).result()
    
    
    
    return(target_table)

