import java.io.*;
import java.util.*;
import java.lang.*;
public class mootel
{
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("mootel.in"));
      int n = in.nextInt(), m = in.nextInt() - 1;
      int[][] a = new int[n][n];
      int done = 1;
      for(int i = 0; i < n; i++)
         for(int j = 0; j < n; j++)
            a[i][j] = in.nextInt();
      in.close();
      boolean[] visited = new boolean[n];
      visited[m] = true;
      Queue<Integer> qn = new ArrayDeque<>();
      Queue<Integer> qm = new ArrayDeque<>();
      ArrayList[] ans = new ArrayList[1000000];
      for(int i = 0; i < ans.length; i++)
         ans[i] = new ArrayList<Integer>();
      
      ans[0].add(m + 1);
      int moves = 1;
      int currentr = m;
      
      for(int i = 0; i < n; i++)
         if(a[m][i] == 1 && !visited[i])
         {
            ans[moves].add(i + 1);
            qn.add(i);
            qm.add(moves);
            currentr = i;
            visited[i] = true;
         }
      
      while(!qn.isEmpty())
      {
         currentr = qn.peek();
         int currentm = qm.peek();
         qn.remove();
         qm.remove();
         for(int i = 0; i < n; i++)
         {
            if(a[currentr][i] == 1 && !visited[i])
            {
               ans[currentm + 1].add(i + 1);
               qn.add(i);
               qm.add(currentm + 1);
               visited[i] = true;
            }
         }   
      }
      
      int i = 0;
      while(!ans[i].isEmpty())
      {
         Collections.sort(ans[i]);
         for(int j = 0; j < ans[i].size(); j++)
            System.out.print(ans[i].get(j) + " ");
         System.out.println();
         i++;
      }
   }
}