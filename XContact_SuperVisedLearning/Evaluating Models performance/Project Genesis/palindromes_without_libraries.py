# To find palindromic substrings in a string using dynamic programming, we can create a solution that builds a table to keep track of which substrings are palindromic.
# This approach has a time complexity of O(n^2), where n is the length of the string. Here’s a Python implementation:

# 1. Initialization: Create a 2D list dp where dp[i][j] will be True if the substring s[i:j+1] is a palindrome.
# 2. Single-character Palindromes: Set each dp[i][i] to True as any single character is a palindrome.
# 3. Two-character Palindromes: Check if consecutive characters are the same; if they are, mark dp[i][i+1] as True.
# 4. Larger Substrings: For substrings longer than 2 characters, set dp[i][j] to True if s[i] == s[j] and the substring s[i+1:j] is also a palindrome.

def find_palindromic_substrings(s):
    n = len(s)
    dp = [[False] * n for _ in range(n)]
    palindromes = []

    # Check for substrings of length 2
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            # Key change for not considering length 1 sub-strings - while also maintaining Logic consistency
            # palindromes.append(s[i:i + 2])

    # Check for substrings of length greater than 2
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                palindromes.append(s[i:j + 1])

    return palindromes


# Example usage
input_string = "racecar"
print("Palindromic substrings:", find_palindromic_substrings(input_string))
