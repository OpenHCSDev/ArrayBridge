# Setup Instructions for Badges and Documentation

This document contains the manual steps needed to complete the badge and documentation setup for arraybridge.

## 1. Enable GitHub Pages

To display the coverage reports hosted on GitHub Pages:

1. Go to: https://github.com/trissim/arraybridge/settings/pages
2. Under "Build and deployment" > "Source", select **"GitHub Actions"**
3. Save the changes

Once enabled, the coverage-pages.yml workflow will deploy coverage reports to https://trissim.github.io/arraybridge/coverage/

## 2. Update Repository Description

To add the ReadTheDocs link to the repository description:

1. Go to: https://github.com/trissim/arraybridge
2. Click the gear icon (⚙️) next to "About" on the right side
3. In the "Website" field, enter: `https://arraybridge.readthedocs.io`
4. Optionally add topics/tags to improve discoverability
5. Click "Save changes"

## 3. Verify ReadTheDocs Integration

Ensure ReadTheDocs is properly configured:

1. Go to: https://readthedocs.org/projects/arraybridge/
2. Verify that the project is active and building successfully
3. Check that the webhook is configured (Settings > Integrations)
4. The `.readthedocs.yml` file is already configured in the repository

## 4. Verify Codecov Integration

The existing CI workflow (`.github/workflows/ci.yml`) already uploads coverage to Codecov:

- Coverage is uploaded after tests run on ubuntu-latest with Python 3.12 and torch framework
- The coverage data is sent to Codecov using the `codecov/codecov-action@v3`
- No additional Codecov configuration is needed

## 5. Test the New Workflow

After merging this PR to main:

1. The `coverage-pages.yml` workflow will run automatically
2. It will:
   - Run tests with coverage
   - Generate a coverage badge and commit it to `.github/badges/coverage.svg`
   - Deploy the HTML coverage report to GitHub Pages
3. The badges in README.md will then display live data

## Badge URLs

The following badges have been added to README.md:

- **ReadTheDocs**: `[![Documentation Status](https://readthedocs.org/projects/arraybridge/badge/?version=latest)](https://arraybridge.readthedocs.io/en/latest/?badge=latest)`
- **Coverage**: `[![Coverage](https://raw.githubusercontent.com/trissim/arraybridge/main/.github/badges/coverage.svg)](https://trissim.github.io/arraybridge/coverage/)`

## Notes

- The coverage badge will show 0% until the first successful run of the coverage-pages.yml workflow on the main branch
- The ReadTheDocs badge will show the build status (passing/failing) based on the latest documentation build
- Both workflows are set up similarly to the ezstitcher repository for consistency
