# Catalog notes and estimation policy

`works.txt` is the preserved source supplied with this project. `data/works.csv` is a
hand-curated, machine-readable derivative containing all 48 rows in that source.
It was built without network access. The source supplies title, date range, and a
short significance note, but generally does not supply artists or exact issue counts.

The derived fields are intentionally estimates, not claims of bibliographic precision:

- `character_prominence` is an editorial 1-10 score for how prominent the featured
  character/team is in mainstream superhero comics: 10 is a flagship property or
  universe-wide ensemble, 7-9 is a major established property, 4-6 is a recognized
  supporting/cult property, and 1-3 is a new, niche, or unspecified property.
- `artist_notability` is an editorial 1-5 career-recognition score for the listed
  representative artist(s): 5 is an industry-leading or especially celebrated name,
  4 is a widely established mainstream artist, 3 is established, and 1-2 is less
  established or unclear. A multi-artist work receives one overall estimate.
- `artist_names` contains representative principal artists remembered from the work,
  not exhaustive credits. This metadata should be verified against published credits
  before being used as a bibliography.
- `issue_count` is the number of issues/volumes in the relevant core work or Johns
  contribution when reasonably clear. It is not the total number of crossover tie-ins.
  `issue_count_basis` states whether the value came directly from the source, is a
  known limited-series length, is approximate, or is deliberately blank.
- Blank numeric cells are real missing values. The model median-imputes them and adds
  a missing-value indicator; it does not silently replace them in the catalog.
- `metadata_confidence` summarizes confidence in the derived bibliographic row, not
  the artistic quality of the work.

The source's `Ghost Machine` row describes a company/publishing initiative rather than
a reading work. It remains in the CSV with `recommendable=false`, so it is preserved
but never presented as something to rate. Announced and ongoing works remain
eligible, but their unknown issue counts and other gaps stay blank.

Personal reading ratings use a separate ordered 1-5 scale: 1 is horrible, 2 is poor,
3 is okay, 4 is great, and 5 is amazing. These personal values are labels for the
recommender; they are unrelated to the catalog's 1-5 artist-notability estimate.

The recommender also turns `character_or_team` into canonical multi-hot tags. It
splits combined credits, normalizes aliases such as Barry Allen/Wally West to a shared
`flash` franchise tag, and retains selected specific identities. This lets preference
signals transfer between related works without pretending that unrelated teams are
the same. The original catalog text remains unchanged and is still shown to users.

Some source rows overlap: for example, `Sinestro Corps War` is part of the longer
`Green Lantern` run, and `Darkseid War` is part of `Justice League`. They remain
separate because the supplied source treats them as separate reading choices.
