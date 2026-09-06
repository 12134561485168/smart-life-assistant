---
name: themealdb
description: Find, verify, and present meal recipes, ingredients, food images, cuisines, categories, and structured recipe data with TheMealDB JSON API. Use when a user asks for a meal recipe, dishes containing an ingredient, cuisine or category ideas, meal planning, ingredient information, or help integrating TheMealDB into an app.
---

# Use TheMealDB

Use TheMealDB as the structured source for meal and ingredient data. Query the
official JSON API instead of scraping website pages.

Base URL:

```text
https://www.themealdb.com/api/json/v1/{API_KEY}/
```

Use development key `1` only for development or educational work. Direct production
and publicly released application users to the current access options at
https://www.themealdb.com/documentation.

## Choose the operation

- Search by meal name: `search.php?s=Arrabiata`
- Search by first letter: `search.php?f=a`
- Filter by one ingredient: `filter.php?i=chicken_breast`
- Filter by category: `filter.php?c=Seafood`
- Filter by area: `filter.php?a=Canadian`
- Look up a complete recipe: `lookup.php?i=52772`
- Get a random meal: `random.php`
- List categories, areas, or ingredients: `list.php?c=list`,
  `list.php?a=list`, or `list.php?i=list`
- List detailed categories: `categories.php`

URL-encode user input. Accept underscores as spaces in documented filter values.
Do not place untrusted input into a URL without encoding it.

## Retrieve a recipe

1. Search or filter for candidate meals.
2. Select a candidate using its `idMeal`.
3. Call `lookup.php?i={idMeal}` to retrieve the complete record. Always perform
   this lookup after a filter because filter results contain summaries rather than
   full recipes.
4. Read `strMeal`, `strCategory`, `strArea`, `strInstructions`, `strMealThumb`,
   `strSource`, and `strYoutube` when present.
5. Pair `strIngredientN` with `strMeasureN` for `N` from 1 through 20. Preserve
   order, trim whitespace, and omit null or empty ingredient slots. Retain an
   ingredient when its measure is blank.
6. Present only data supported by the returned record. Never invent a measure,
   ingredient, dietary property, or preparation step.

Treat an empty array, null value, or `no data found` response as not found. Try a
spelling correction or broader search when appropriate, and disclose that choice.

## Answer well

For a recipe request, include:

- meal name linked to `https://www.themealdb.com/meal/{idMeal}`;
- category and area/cuisine when present;
- ingredients paired with their measures;
- preparation instructions;
- source or video links when useful;
- a source credit to TheMealDB.

For an ingredient-on-hand request, look up every suggested meal before answering.
Compare its complete ingredient list with the user's available ingredients and
clearly distinguish exact matches from recipes that require additional items.

For comparisons or meal plans, use consistent fields across meals and avoid
claiming a filter response is exhaustive when the user's access tier may limit
result counts.

Suggested credit:

```text
Recipe data and imagery: TheMealDB (https://www.themealdb.com/)
```

## Use images

Prefer the `strMealThumb` URL returned by the API. Append `/small`, `/medium`, or
`/large` for 200, 350, or 500 pixel meal variants.

Build an ingredient image URL only when needed:

```text
https://www.themealdb.com/images/ingredients/{URL_ENCODED_NAME}.png
```

Append `/small`, `/medium`, or `/large` for an ingredient size variant. Preserve
attribution and any licensing metadata in the record.

## Apply safety and policy

- Do not infer that a recipe is allergen-free, vegetarian, vegan, halal, kosher,
  or medically suitable from its name, tags, category, or area alone.
- Treat allergy questions as high risk. Explain that TheMealDB is not an
  allergen-certification service and advise checking ingredient labels and
  cross-contamination risks.
- Do not invent cooking temperatures, storage advice, or food-safety claims.
- Use official API endpoints rather than scraping the website.
- Do not resell the API or remove copyright and trademark notices.
- Check current production, commercial-use, and artwork terms before publication:
  https://www.themealdb.com/terms_of_use.php

For expanded API details and agent-oriented field guidance, read
https://www.themealdb.com/AGENTS.md.
