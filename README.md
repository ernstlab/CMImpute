# CMImpute
Cross-species and tissue imputation of species-level DNA methylation samples across mammalian species.

CMImpute (Cross-species Methylation Imputation) is a imputation method based on a Conditional Variational Autoencoder (CVAE) to impute methylation values for species-tissue combinations that have not previously been experimentally profiled. CMImpute takes as input individual methylation samples along with their corresponding species and tissue labels. CMImpute outs a species-tissue combination mean sample, or combination mean sample for short, that represents a species' average methylation values in a particular tissue type.

<img src='method_overview_main_fig.png' width='512'>

## Precomputed combination mean samples

Final imputed species-tissue combination mean samples for species and tissue combinations that have not been experimentally profiled (19,786 magenta imputed combinations in figure above) can be found <a href="https://public.hoffman2.idre.ucla.edu/ernst/0TDK7/">here</a>.

## Dependencies
- keras 2.10.0
- numpy 1.23.4
- pandas 1.4.4
- scikit-learn 0.24.2
- scipy 1.9.3
- tensorflow 2.10.0

## Data Format
The training data input should be formatted as such.
- First column containing row names
- First row containing species and tissue names folowed by probe name
- Data consisting of one-hot encoded species and tissue labels followed by methylation values
<img src='example_input.png'>
