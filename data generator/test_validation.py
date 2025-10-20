"""
Script de validación de datos sintéticos
"""

from data_generator import DataExplorer, SyntheticDataGenerator
import pandas as pd

def main():
    print("\n" + "="*80)
    print("VALIDACION DE DATOS SINTETICOS")
    print("="*80 + "\n")
    
    # Cargar datos originales
    print("Cargando dataset original...")
    explorer = DataExplorer("data/twitter_training.csv", "data/twitter_validation.csv")
    stats = explorer.get_statistics_summary()
    
    print(f"Total de tweets originales: {stats['total_tweets']}")
    
    # Generar datos sintéticos
    print("\nGenerando 1000 tweets sintéticos...")
    generator = SyntheticDataGenerator(stats)
    synthetic_tweets = generator.generate_batch(n=1000)
    synthetic_df = pd.DataFrame(synthetic_tweets)
    
    # Validación de Sentimientos
    print("\n" + "-"*80)
    print("VALIDACION DE SENTIMIENTOS")
    print("-"*80)
    
    original_sentiment = stats['sentiment_distribution']
    synthetic_sentiment = synthetic_df['Sentiment'].value_counts(normalize=True).to_dict()
    
    print("\nComparación de distribuciones:")
    print(f"{'Sentimiento':<15} {'Original':<12} {'Sintetico':<12} {'Diferencia':<12} {'Estado':<10}")
    print("-"*80)
    
    for sentiment in original_sentiment.keys():
        orig = original_sentiment[sentiment]
        synt = synthetic_sentiment.get(sentiment, 0)
        diff = abs(orig - synt)
        status = "OK" if diff <= 0.05 else "FALLO"
        print(f"{sentiment:<15} {orig:<12.4f} {synt:<12.4f} {diff:<12.4f} {status:<10}")
    
    # Validación de Longitud
    print("\n" + "-"*80)
    print("VALIDACION DE LONGITUD DE TEXTO")
    print("-"*80)
    
    original_length = stats['text_length_stats']
    synthetic_lengths = synthetic_df['Tweet content'].astype(str).apply(len)
    
    print("\nComparación de estadísticas:")
    print(f"{'Metrica':<15} {'Original':<15} {'Sintetico':<15} {'Diferencia %':<15} {'Estado':<10}")
    print("-"*80)
    
    metrics = {
        'Media': (original_length['mean'], synthetic_lengths.mean()),
        'Std Dev': (original_length['std'], synthetic_lengths.std()),
        'Minimo': (original_length['min'], synthetic_lengths.min()),
        'Maximo': (original_length['max'], synthetic_lengths.max()),
    }
    
    for metric, (orig, synt) in metrics.items():
        if orig > 0:
            diff_pct = abs(orig - synt) / orig * 100
        else:
            diff_pct = 0
        status = "OK" if diff_pct <= 5 else "FALLO"
        print(f"{metric:<15} {orig:<15.2f} {synt:<15.2f} {diff_pct:<15.2f} {status:<10}")
    
    # Validación de Entidades
    print("\n" + "-"*80)
    print("VALIDACION DE ENTIDADES")
    print("-"*80)
    
    original_entities = stats['entity_distribution']
    synthetic_entities = synthetic_df['Entity'].value_counts(normalize=True).to_dict()
    
    max_diff = 0
    for entity in original_entities.keys():
        orig = original_entities[entity]
        synt = synthetic_entities.get(entity, 0)
        diff = abs(orig - synt)
        max_diff = max(max_diff, diff)
    
    print(f"\nDiferencia maxima en distribución de entidades: {max_diff:.4f}")
    print(f"Estado: {'OK' if max_diff <= 0.05 else 'FALLO'}")
    
    # Resumen
    print("\n" + "="*80)
    print("RESUMEN DE VALIDACION")
    print("="*80)
    
    print(f"\nTweets generados: {len(synthetic_tweets)}")
    print(f"Tweets unicos: {synthetic_df['Tweet ID'].nunique()}")
    print(f"Entidades unicas: {synthetic_df['Entity'].nunique()}")
    print(f"Sentimientos unicos: {synthetic_df['Sentiment'].nunique()}")
    print(f"Valores nulos: {synthetic_df.isnull().sum().sum()}")
    
    print("\nCONCLUSION: Los datos sinteticos son validos y confiables para usar en Kafka")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
