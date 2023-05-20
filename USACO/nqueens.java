import java.io.*;
import java.util.*;
import java.lang.*;
public class nqueens
{
   static int[] queens;
   static int n;
   static int sol = 0;
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("nqueens.in"));
      n = in.nextInt();
      in.close();
      
      queens = new int[n];
      sol = dfs(0);
      System.out.println(sol);
   }
   
   public static int dfs(int pos)
   {
      if(pos == n) return 1;
       
      int ans = 0;
      for(int i = 0; i < n; i++)
         if(isOk(pos, i))
         {
            queens[pos] = i;
            ans += dfs(pos + 1);
         }
        
      return ans;
   }
   
   public static boolean isOk(int pos, int i)
   {
      for(int k = 0; k < pos; k++)
      {
         if(i == queens[k]) 
            return false;
         if(i + pos == queens[k] + k) 
            return false;
         if(i - pos == queens[k] - k) 
            return false;
      }
        
      return true;
   }
}