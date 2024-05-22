import os
import random
import re
import sys

DAMPING = 0.85  # Damping factor used in PageRank calculation
SAMPLES = 10000  # Number of samples for the sampling method

def main():
    # Check if the correct number of arguments are provided
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    # Extract links from HTML pages in the provided directory
    corpus = crawl(sys.argv[1])
    # Compute PageRank using sampling method
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    # Print PageRank results
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    # Compute PageRank using iteration method
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    # Print PageRank results
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            # Store links excluding self-references
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages

def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    num_pages = len(corpus)
    probabilities = {}
    
    # If the page has no outgoing links, treat it as having links to all pages
    if not corpus[page]:
        for p in corpus:
            probabilities[p] = 1 / num_pages
        return probabilities

    # Calculate the probability distribution
    for p in corpus:
        probabilities[p] = (1 - damping_factor) / num_pages
        if p in corpus[page]:
            probabilities[p] += damping_factor / len(corpus[page])

    return probabilities

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initialize PageRank values for each page
    pagerank = {page: 0 for page in corpus}

    # Choose a random page to start
    current_page = random.choice(list(corpus.keys()))

    # Perform sampling for n iterations
    for _ in range(n):
        # Update the PageRank value of the current page
        pagerank[current_page] += 1

        # Transition to a new page based on the transition model
        model = transition_model(corpus, current_page, damping_factor)
        current_page = random.choices(list(model.keys()), weights=model.values(), k=1)[0]

    # Normalize PageRank values so they sum to 1
    total_samples = sum(pagerank.values())
    pagerank = {page: count / total_samples for page, count in pagerank.items()}

    return pagerank

def iterate_pagerank(corpus, damping_factor, epsilon=1e-8):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    num_pages = len(corpus)
    
    # Initialize PageRank values for each page
    pagerank = {page: 1 / num_pages for page in corpus}

    while True:
        new_pagerank = {}
        for page in corpus:
            # Calculate the sum of PageRank values of pages that link to the current page
            incoming_pagerank = sum(pagerank[from_page] / len(corpus[from_page]) for from_page in corpus if page in corpus[from_page])
            # Update the PageRank value for the current page
            new_pagerank[page] = (1 - damping_factor) / num_pages + damping_factor * incoming_pagerank

        # Check for convergence
        if all(abs(new_pagerank[page] - pagerank[page]) < epsilon for page in corpus):
            break

        # Update PageRank values for the next iteration
        pagerank = new_pagerank

    return pagerank

if __name__ == "__main__":
    main()
