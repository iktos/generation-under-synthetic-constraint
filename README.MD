# Package description

This package is a fork of the code of BenevolantAI https://github.com/BenevolentAI/guacamol_baselines.
It enables running generations with several models, for the tasks from the guacamol benchmark ,
and for another task named "pi3kmtor" described in the paper.

Additionaly, the package enables to run generations with synthetic score as a new constraint.
The synthetic accessibility score can be one of the following :

- SA score
- RA score
- SC score
- RScore (Spaya Api)
- RSPred (Predictor of SpayaApi trained on Chembl24).

# Requirements

- Run `poetry install` to install all the required dependencies.

- Download the guacamol SMILES files with the script found in `guacamol_baselines/fetch_guacamol_dataset.sh`
- Extract the `models.zip` file found in the folder `synthetic_scorers/RAscore`
- Generated smiles are stored in a MongoDB database, so you need to setup a database and set the following environment variables to point to it:
  - `MONGO_URL`: URI and necessary credentials to the database server
  - `DB_STORAGE`: Name of the database to be used to store the sampled SMILES (the code will create collections to store the data)
- Finally, if you want to use the RScore score, calculated by the Spaya API, you need to have two
  more environment variables:
  - `SPAYA_API_URL`
  - `SPAYA_API_TOKEN`

These should contain your credentials to use the Spaya API.

# Run generations

The goal directed generations have 2 essentials arguments :

- **suite**: this value will determine the generations that are going to be launched. A value of a
  suite correponds to a list of tasks:
  - **'guacamol_paper'** launches the generation of
    guacamol benchamrk tasks with successively no synthetic scores, SA score, RScore and RSPred.
  - **'pi3kmtor_paper'** launches the pi3kmtor generation with successively each of the
    synthetic scorers.
  - **'pi3kmtor'** launches only one generation of pi3kmtor, using the synthetic score set in the `synth_score` variable (see below)
- **synth_score** : when set, indicates which synthetic scorer to add to the generation. Should
  be one of `RScore, SAscore, SCscore, RAscore, or RSPred`. This parameter is only used by the `pi3kmtor` suite.

## Example

Run 10 steps of generation around pi3kmtor dataset, optimizing 4 constraints and using the `SA score` constraint:

`poetry run python -m guacamol_baselines.smiles_lstm_hc.goal_directed_generation --suite pi3kmtor --n_epochs 10 --synth_score SAscore `

# Get the results

- The generated molecules are saved in the MongoDB database defined above, in collections named with the `synth_score` used by each task (eg. `benchmark_name+"_"+<synth_score>`)

- For the pi3kmtor generations, the differents scores of the molecules are already in the collection.

- You can use the notebook in `exploit_results/exploit_results_pi3kmtor.ipynb` to analyse the results.
