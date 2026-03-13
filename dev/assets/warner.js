function maybeAddWarning() {
  // DOCUMENTER_NEWEST is defined in versions.js, DOCUMENTER_CURRENT_VERSION and DOCUMENTER_STABLE
  // in siteinfo.js. DOCUMENTER_IS_DEV_VERSION is optional and defined in siteinfo.js.
  // If the required variables are undefined something went horribly wrong, so we abort.
  if (
    window.DOCUMENTER_NEWEST === undefined ||
    window.DOCUMENTER_CURRENT_VERSION === undefined ||
    window.DOCUMENTER_STABLE === undefined
  ) {
    return;
  }

  // Current version is not a version number, so we can't tell if it's the newest version. Abort.
  if (!/v(\d+\.)*\d+/.test(window.DOCUMENTER_CURRENT_VERSION)) {
    return;
  }

  // Current version is newest version, so no need to add a warning.
  if (window.DOCUMENTER_NEWEST === window.DOCUMENTER_CURRENT_VERSION) {
    return;
  }

  // Add a noindex meta tag (unless one exists) so that search engines don't index this version of the docs.
  if (document.body.querySelector('meta[name="robots"]') === null) {
    const meta = document.createElement("meta");
    meta.name = "robots";
    meta.content = "noindex";

    document.getElementsByTagName("head")[0].appendChild(meta);
  }

  const div = document.createElement("div");
  // Base class is added by default
  div.classList.add("warning-overlay-base");
  const closer = document.createElement("button");
  closer.classList.add("outdated-warning-closer", "delete");
  closer.addEventListener("click", function () {
    document.body.removeChild(div);
  });
  var target_href =
    window.documenterBaseURL + "/../" + window.DOCUMENTER_STABLE;

  // try to stay on the same page when linking to the stable version
  // get the current page path relative to the version root
  var current_page = window.location.pathname;

  // resolve the documenterBaseURL to an absolute path
  // documenterBaseURL is a relative path (usually "."), so we need to resolve it
  var base_url_absolute = new URL(documenterBaseURL, window.location.href)
    .pathname;
  if (!base_url_absolute.endsWith("/")) {
    base_url_absolute = base_url_absolute + "/";
  }

  // extract the page path after the version directory
  // e.g., if we're on /stable/man/guide.html, we want "man/guide.html"
  var page_path = "";
  if (current_page.startsWith(base_url_absolute)) {
    page_path = current_page.substring(base_url_absolute.length);
  }

  // construct the target URL with the same page path
  var target_url = target_href;
  if (page_path && page_path !== "" && page_path !== "index.html") {
    // ensure target_href ends with a slash before appending page path
    if (!target_url.endsWith("/")) {
      target_url = target_url + "/";
    }
    target_url = target_url + page_path;
  }

  // preserve the anchor (hash) from the current page
  var current_hash = window.location.hash;

  // Determine if this is a development version or an older release
  let warningMessage = "";
  if (window.DOCUMENTER_IS_DEV_VERSION === true) {
    div.classList.add("dev-warning-overlay");
    warningMessage =
      "This documentation is for the <strong>development version</strong> and may contain unstable or unreleased features.<br>";
  } else {
    div.classList.add("outdated-warning-overlay");
    warningMessage =
      "This documentation is for an <strong>older version</strong> that may be missing recent changes.<br>";
  }

  // Create the link element with same-page navigation
  const link = document.createElement("a");
  link.href = target_url + current_hash;
  link.textContent =
    "Click here to go to the documentation for the latest stable release.";

  // If we're trying to stay on the same page, verify it exists first
  if (page_path && page_path !== "" && page_path !== "index.html") {
    link.addEventListener("click", function (e) {
      e.preventDefault();
      // check if the target page exists, fallback to homepage if it doesn't
      fetch(target_url, { method: "HEAD" })
        .then(function (response) {
          if (response.ok) {
            window.location.href = target_url + current_hash;
          } else {
            // page doesn't exist in the target version, go to homepage
            window.location.href = target_href;
          }
        })
        .catch(function (error) {
          // network error or other failure - use homepage
          window.location.href = target_href;
        });
    });
  }

  div.innerHTML = warningMessage;
  div.appendChild(link);
  div.appendChild(closer);
  document.body.appendChild(div);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", maybeAddWarning);
} else {
  maybeAddWarning();
}
